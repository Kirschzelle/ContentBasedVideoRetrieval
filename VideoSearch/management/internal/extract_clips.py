from VideoSearch.management.base import StyledCommand as BaseCommand
from scipy.signal import argrelextrema
import numpy as np
from multiprocessing import Pool, cpu_count
import multiprocessing
from functools import partial
import subprocess
from tqdm import tqdm

DEFAULT_CLIP_EXTRACTION_SETTINGS = {
    "threshold_low": 0.35,
    "threshold_high": 0.99,
    "order_low": 1,
    "max_pass_seconds": 1.5,
    "passes": 25,
}

class Command(BaseCommand):
    pass

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
            # Run sequentially with progress bar
            with tqdm(total=len(video_ids), desc="Extracting clips", unit="video", 
                     dynamic_ncols=True, leave=True, ascii=True) as pbar:
                for i, vid in enumerate(video_ids, 1):
                    result = process_video_for_clips(vid, kwargs, progress=(i, len(video_ids)), pbar=pbar)
                    if result:
                        # Clean up the result message for display
                        clean_result = result.split("] ", 1)[-1] if "] " in result else result
                        pbar.set_postfix_str(clean_result[:50] + "..." if len(clean_result) > 50 else clean_result)
                    pbar.update(1)
        else:
            # Add progress tracking to multiprocessing
            video_data = [(vid, kwargs, (i, len(video_ids))) for i, vid in enumerate(video_ids, 1)]
            with Pool(processes=num_workers) as pool:
                results = pool.starmap(process_video_for_clips_with_progress, video_data)
            for msg in results:
                self.stdout.write(self.style_success(msg))

def process_video_for_clips(video_id, kwargs, progress=None, pbar=None):
    import os
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ContentBasedVideoRetrieval.settings")
    import django
    django.setup()

    from VideoSearch.models import Video, Clip, ClipPredictionCache
    from third_party.transnetv2.inference.transnetv2 import TransNetV2
    from pathlib import Path

    video = Video.objects.get(id=video_id)
    path = Path(video.processing_path)
    
    # Add progress prefix
    progress_str = f"[{progress[0]}/{progress[1]}] " if progress else ""

    existing_clips = Clip.objects.filter(video=video)
    if existing_clips.exists() and is_clip_coverage_complete(video):
        return f"{progress_str}Skipping {path.name} - clips fully exist."

    Clip.objects.filter(video=video).delete()

    model = TransNetV2()
    clips, predictions = extract_clips(model, video, path, pbar=pbar, **kwargs)

    for start_frame, end_frame in clips:
        clip = Clip.objects.create(
            video=video,
            start_frame=start_frame,
            end_frame=end_frame
        )
        ClipPredictionCache.store(clip, predictions[start_frame:end_frame + 1])

    return f"{progress_str}Stored {len(clips)} clips for {path.name}"

def process_video_for_clips_with_progress(video_id, kwargs, progress):
    """Wrapper for multiprocessing that unpacks progress tuple."""
    return process_video_for_clips(video_id, kwargs, progress)
            
def is_clip_coverage_complete(video):
    from VideoSearch.models import Clip
    clips = Clip.objects.filter(video=video).order_by('start_frame')
    if not clips.exists():
        return False

    current = 0
    for clip in clips:
        if clip.start_frame != current:
            print(f"WARNING: Gap in coverage for video {video.id}: expected frame {current}, got {clip.start_frame}")
            return False
        current = clip.end_frame + 1

    if current < video.frame_count:
        print(f"WARNING: Incomplete coverage for video {video.id}: clips end at {current}, video has {video.frame_count} frames")
        return False
    
    return True

def extract_clips(model, video, video_path, pbar=None, **kwargs):
    """
    Runs clip boundary detection on a video file using TransNetV2 predictions.
    Applies multi-pass local maxima analysis with confidence-based filtering.

    Returns: List of (start_frame, end_frame) tuples
    """
    _, single_frame_predictions, _ = model.predict_video(str(video_path))

    fps = video.fps()
    
    # Debug: Check prediction vs frame count mismatch
    pred_len = len(single_frame_predictions)
    if pred_len != video.frame_count:
        if pbar:
            tqdm.write(f"[WARNING] Prediction length ({pred_len}) != video frame count ({video.frame_count}) for {video_path.name}")
            tqdm.write(f"[INFO] Verifying actual frame count...")
        else:
            print(f"WARNING: Prediction length ({pred_len}) != video frame count ({video.frame_count}) for {video_path.name}")
            print(f"Verifying actual frame count...")
        
        # Get actual frame count from video file
        actual_frame_count = get_actual_frame_count(video_path, pbar)
        
        if actual_frame_count is not None:
            if actual_frame_count == pred_len:
                if pbar:
                    tqdm.write(f"[INFO] TransNetV2 is correct ({pred_len} frames), updating video metadata")
                else:
                    print(f"INFO: TransNetV2 is correct ({pred_len} frames), updating video metadata")
                video.frame_count = pred_len
                video.save()
            elif actual_frame_count == video.frame_count:
                if pbar:
                    tqdm.write(f"[INFO] Video metadata is correct ({video.frame_count} frames), TransNetV2 processing incomplete")
                else:
                    print(f"INFO: Video metadata is correct ({video.frame_count} frames), TransNetV2 processing incomplete")
            else:
                if pbar:
                    tqdm.write(f"[INFO] All three counts differ - Metadata: {video.frame_count}, TransNetV2: {pred_len}, Actual: {actual_frame_count}")
                    tqdm.write(f"[INFO] Using actual frame count ({actual_frame_count}) as ground truth")
                else:
                    print(f"INFO: All three counts differ - Metadata: {video.frame_count}, TransNetV2: {pred_len}, Actual: {actual_frame_count}")
                    print(f"Using actual frame count ({actual_frame_count}) as ground truth")
                video.frame_count = actual_frame_count
                video.save()
        else:
            if pbar:
                tqdm.write(f"[WARNING] Could not verify actual frame count, proceeding with existing logic")
            else:
                print(f"WARNING: Could not verify actual frame count, proceeding with existing logic")

    settings = {k: kwargs.get(k, v) for k, v in DEFAULT_CLIP_EXTRACTION_SETTINGS.items()}

    # Set pbar as function attribute for debug messages
    multipass_predictions_to_scenes._pbar = pbar
    
    scenes = multipass_predictions_to_scenes(
        predictions=single_frame_predictions,
        fps=fps,
        **settings,
    )

    # Fix frame count mismatches: handle both under-processing and over-processing
    clips = [(int(start), int(end)) for start, end in scenes]
    if clips and pred_len != video.frame_count:
        # Get actual frame count to ensure we don't create invalid clips
        actual_frame_count = get_actual_frame_count(video_path, pbar)
        if actual_frame_count is None:
            actual_frame_count = pred_len  # Fallback to TransNetV2 count if verification fails
        
        # Use the smaller of database count or actual count to be safe
        safe_max_frame = min(video.frame_count - 1, actual_frame_count - 1)
        
        # Always cap clips to safe bounds, regardless of over/under processing
        last_start, last_end = clips[-1]
        if last_end > safe_max_frame:
            clips[-1] = (last_start, safe_max_frame)
            frame_gap = video.frame_count - pred_len
            if pbar:
                if frame_gap < 0:
                    tqdm.write(f"[INFO] Capped last clip from {last_end} to {safe_max_frame} (TransNetV2 over-processed by {abs(frame_gap)} frames)")
                else:
                    tqdm.write(f"[INFO] Extended last clip by {frame_gap} frames to frame {safe_max_frame} (verified)")
            else:
                if frame_gap < 0:
                    print(f"INFO: Capped last clip from {last_end} to {safe_max_frame} (TransNetV2 over-processed by {abs(frame_gap)} frames)")
                else:
                    print(f"INFO: Extended last clip by {frame_gap} frames to frame {safe_max_frame} (verified)")

    return clips, single_frame_predictions

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
        # Make sure the last scene covers all remaining frames
        scenes.append((start, len(predictions) - 1))
    
    # Debug: Print scene coverage info
    if scenes:
        total_frames = sum(end - start + 1 for start, end in scenes)
        pbar = getattr(multipass_predictions_to_scenes, '_pbar', None)
        if pbar:
            tqdm.write(f"[DEBUG] Created {len(scenes)} scenes covering {total_frames} frames, predictions length: {len(predictions)}")
        else:
            print(f"DEBUG: Created {len(scenes)} scenes covering {total_frames} frames, predictions length: {len(predictions)}")

    return scenes

def get_actual_frame_count(video_path, pbar=None):
    """
    Get the actual number of decodable frames in a video file.
    More accurate than metadata-based estimates.
    """
    try:
        # Count actual decodable frames using ffprobe
        cmd = [
            "ffprobe", 
            "-v", "quiet",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse the frame count directly
        frame_count = int(result.stdout.strip())
        
        return frame_count if frame_count > 0 else None
        
    except (subprocess.CalledProcessError, Exception) as e:
        if pbar:
            tqdm.write(f"[ERROR] Could not get actual frame count for {video_path}: {e}")
        else:
            print(f"ERROR: Could not get actual frame count for {video_path}: {e}")
        return None