from VideoSearch.management.base import StyledCommand as BaseCommand
from VideoSearch.models import Video, Clip, ClipPredictionCache
from third_party.transnetv2.inference.transnetv2 import TransNetV2
from scipy.signal import argrelextrema
from pathlib import Path
import numpy as np

DEFAULT_CLIP_EXTRACTION_SETTINGS = {
    "threshold_low": 0.5,
    "threshold_high": 0.99,
    "order_low": 1,
    "max_pass_seconds": 1.0,
    "passes": 10,
}

class Command(BaseCommand):
    help = "Extract clips from all videos and store them in the database.\nA clip is defined as a single Video sequence with no cuts within it."

    def add_arguments(self, parser):
        for key, default in DEFAULT_CLIP_EXTRACTION_SETTINGS.items():
            arg_name = f"--{key.replace('_', '-')}"
            arg_type = float if isinstance(default, float) else int

            if key == "passes":
                parser.add_argument(arg_name, type=arg_type, default=default, choices=range(1, 21),
                                    help="Number of detection passes (1-20)")
            else:
                parser.add_argument(arg_name, type=arg_type, default=default)

    def handle(self, *args, **kwargs):

        videos = Video.objects.all()
        model = TransNetV2()

        for video in videos:
            path = Path(video.file_path)
            existing_clips = Clip.objects.filter(video=video)
            if existing_clips.exists():
                if is_clip_coverage_complete(video):
                    self.stdout.write(self.style_info(f"Skipping {path.name} - clips fully exist."))
                    continue
                else:
                    self.stdout.write(self.style_warning(f"Incomplete clips for {path.name} - recalculating."))
                    Clip.objects.filter(video=video).delete()

            self.stdout.write(self.style_info(f"Processing {path.name}"))

            clips, predictions = extract_clips(model, video, path, **kwargs)

            for start_frame, end_frame in clips:
                clip = Clip.objects.create(
                    video=video,
                    start_frame=start_frame,
                    end_frame=end_frame
                )
                ClipPredictionCache.store(clip, predictions[start_frame:end_frame + 1])

            self.stdout.write(self.style_success(f"Stored {len(clips)} clips for {path.name}"))
            
def is_clip_coverage_complete(video):
    clips = Clip.objects.filter(video=video).order_by('start_frame')
    if not clips.exists():
        return False

    current = 0
    for clip in clips:
        if clip.start_frame != current:
            return False
        current = clip.end_frame + 1

    return current >= video.frame_count

def extract_clips(model, video, video_path, **kwargs):
    """
    Runs clip boundary detection on a video file using TransNetV2 predictions.
    Applies multi-pass local maxima analysis with confidence-based filtering.

    Returns: List of (start_frame, end_frame) tuples
    """
    _, single_frame_predictions, _ = model.predict_video(str(video_path))

    fps = video.fps()

    settings = {k: kwargs.get(k, v) for k, v in DEFAULT_CLIP_EXTRACTION_SETTINGS.items()}

    scenes = multipass_predictions_to_scenes(
        predictions=single_frame_predictions,
        fps=fps,
        **settings,
    )

    return [(int(start), int(end)) for start, end in scenes], single_frame_predictions

def multipass_predictions_to_scenes(
    predictions: np.ndarray,
    threshold_low: float,
    threshold_high: float,
    order_low: int,
    max_pass_seconds: float,
    passes: int,
    fps: float
):
    """
    Runs multiple passes of local maxima detection with decreasing thresholds and increasing smoothing.
    
    :param predictions: 1D numpy array of shot boundary confidences per frame.
    :param threshold_low: Minimum threshold value to consider.
    :param threshold_high: Starting threshold (usually 1.0).
    :param order_low: Starting order for sharp cuts (usually 3).
    :param max_pass_seconds: Maximum temporal smoothing in seconds for final pass.
    :param passes: Number of passes to run.
    :param fps: Frames per second.
    :return: List of (start_frame, end_frame) tuples.
    """
    predictions = np.asarray(predictions)
    cut_candidates = set()

    order_high = int(max_pass_seconds * fps / 2)

    for i in range(passes):
        t = i / (passes - 1) if passes > 1 else 0
        threshold = threshold_high - t * (threshold_high - threshold_low)
        order = max(1, int(order_low + t * (order_high - order_low)))

        local_maxima = argrelextrema(predictions, np.greater, order=order)[0]
        new_cuts = [idx for idx in local_maxima if predictions[idx] > threshold]
        cut_candidates.update(new_cuts)

    sorted_cuts = sorted(cut_candidates)

    if not sorted_cuts:
        return [(0, len(predictions) - 1)]

    scenes = []
    start = 0
    for cut in sorted_cuts:
        scenes.append((start, cut))
        start = cut + 1

    if start < len(predictions):
        scenes.append((start, len(predictions) - 1))

    return scenes