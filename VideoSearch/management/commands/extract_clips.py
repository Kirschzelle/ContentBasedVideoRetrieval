from VideoSearch.management.base import StyledCommand as BaseCommand
from scipy.signal import argrelextrema
import numpy as np
from multiprocessing import Pool, cpu_count
import multiprocessing
from functools import partial

DEFAULT_CLIP_EXTRACTION_SETTINGS = {
    "threshold_low": 0.45,
    "threshold_high": 0.99,
    "order_low": 1,
    "max_pass_seconds": 0.3,
    "passes": 20,
}

class Command(BaseCommand):
    help = "Extract clips from all videos and store them in the database.\nA clip is defined as a single Video sequence with no cuts within it."

    def add_arguments(self, parser):
        parser.add_argument('--workers', type=int, default=4, help='Number of worker processes (1 disables multiprocessing).')
        for key, default in DEFAULT_CLIP_EXTRACTION_SETTINGS.items():
            arg_name = f"--{key.replace('_', '-')}"
            arg_type = float if isinstance(default, float) else int

            if key == "passes":
                parser.add_argument(arg_name, type=arg_type, default=default, choices=range(1, 101),
                                    help="Number of detection passes (1-100)")
            else:
                parser.add_argument(arg_name, type=arg_type, default=default)

    def handle(self, *args, **kwargs):
        multiprocessing.set_start_method('spawn', force=True)

        from multiprocessing import freeze_support
        freeze_support()

        self._handle_multiprocess(**kwargs)

    def _handle_multiprocess(self, **kwargs):
        import os
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ContentBasedVideoRetrieval.settings")
        import django
        django.setup()

        from VideoSearch.models import Video

        videos = Video.objects.all()
        video_ids = [v.id for v in videos]

        num_workers = kwargs.get("workers", 4)
        self.stdout.write(self.style_info(f"Processing {len(video_ids)} videos using {num_workers} worker(s)."))

        if num_workers == 1:
            # Run sequentially (no Pool)
            results = [process_video_for_clips(vid, kwargs) for vid in video_ids]
        else:
            with Pool(processes=num_workers) as pool:
                results = pool.map(partial(process_video_for_clips, kwargs=kwargs), video_ids)

        for msg in results:
            self.stdout.write(self.style_success(msg))

def process_video_for_clips(video_id, kwargs):
    import os
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ContentBasedVideoRetrieval.settings")
    import django
    django.setup()

    from VideoSearch.models import Video, Clip, ClipPredictionCache
    from third_party.transnetv2.inference.transnetv2 import TransNetV2
    from pathlib import Path

    video = Video.objects.get(id=video_id)
    path = Path(video.file_path)

    existing_clips = Clip.objects.filter(video=video)
    if existing_clips.exists() and is_clip_coverage_complete(video):
        return f"Skipping {path.name} - clips fully exist."

    Clip.objects.filter(video=video).delete()

    model = TransNetV2()
    clips, predictions = extract_clips(model, video, path, **kwargs)

    for start_frame, end_frame in clips:
        clip = Clip.objects.create(
            video=video,
            start_frame=start_frame,
            end_frame=end_frame
        )
        ClipPredictionCache.store(clip, predictions[start_frame:end_frame + 1])

    return f"Stored {len(clips)} clips for {path.name}"
            
def is_clip_coverage_complete(video):
    from VideoSearch.models import Clip
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