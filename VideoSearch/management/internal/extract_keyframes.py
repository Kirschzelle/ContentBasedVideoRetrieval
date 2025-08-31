from VideoSearch.management.base import StyledCommand as BaseCommand
from VideoSearch.management.internal.extract_clips import multipass_predictions_to_scenes
from multiprocessing import Pool
from tqdm import tqdm
import math

feature_extractor = None

def calculate_frames_to_compare(region_seconds):
    """
    Calculate frames to sample per region based on region length using smooth curve.
    Short regions (2s): ~12 frames, Long regions (60s): ~50 frames
    """
    base = 5.5  # Higher base for better short region coverage
    sqrt_component = math.sqrt(region_seconds) * base
    linear_component = region_seconds * 0.4
    
    frames = int(sqrt_component + linear_component)
    
    # More generous minimum, especially for short regions
    return max(12, min(50, frames))

class Command(BaseCommand):
    pass

    def add_arguments(self, parser):
        parser.add_argument('--threshold', type=float, default=0.35, help='Distance threshold for keyframe uniqueness.')
        parser.add_argument('--search-range-factor', type=float, default=0.95, help='Fraction of region used to search around potential keyframe.')
        parser.add_argument('--frames-to-compare', type=int, default=25, help='How many frames to sample when searching for keyframes.')
        parser.add_argument('--workers', type=int, default=4, help='Number of worker processes (1 disables multiprocessing).')

    def handle(self, *args, **kwargs):
        from VideoSearch.models import ClipPredictionCache
        threshold = kwargs.get('threshold', 0.35)
        search_range_factor = kwargs.get('search_range_factor', 0.95)
        frames_to_compare = kwargs.get('frames_to_compare', 25)
        workers = kwargs.get('workers', 4)

        candidates = ClipPredictionCache.objects.select_related("clip", "clip__video").all()

        if not candidates:
            self.stdout.write(self.style_warning("No clips require keyframe extraction."))
            return

        args_list = [
            (entry.id, threshold, search_range_factor, frames_to_compare, (i, len(candidates)))
            for i, entry in enumerate(candidates, 1)
        ]

        self.stdout.write(self.style_info(f"Extracting keyframes for {len(candidates)} clips using {workers} worker(s)."))

        if workers == 1:
            init_worker()
            with tqdm(total=len(args_list), desc="Extracting keyframes", unit="clip", 
                     dynamic_ncols=True, leave=True, ascii=True) as pbar:
                for i, args in enumerate(args_list, 1):
                    # Show what we're about to process
                    entry_id = args[0]  # First arg is entry_id
                    pbar.set_postfix_str(f"Starting clip {i}/{len(args_list)} (ID: {entry_id})")
                    pbar.refresh()
                    
                    # Add pbar to args for debug message handling
                    args_with_pbar = args + (pbar,)
                    
                    try:
                        result = process_clip_entry_worker(*args_with_pbar)
                        
                        # Update with result
                        if result:
                            clip_info = result.split("] ", 1)[-1] if "] " in result else result
                            pbar.set_postfix_str(clip_info[:60])  # Truncate long messages
                        
                    except Exception as e:
                        pbar.set_postfix_str(f"ERROR on clip {entry_id}: {str(e)[:40]}")
                        # Continue processing other clips even if one fails
                    
                    pbar.update(1)
                    pbar.refresh()
        else:
            with Pool(processes=workers, initializer=init_worker) as pool:
                results = pool.starmap(process_clip_entry_worker, args_list)
            for result in results:
                if result:
                    self.stdout.write(self.style_success(result))

def process_clip_entry(entry, feature_extractor, threshold, search_range_factor, frames_to_compare, progress=None, pbar=None, command=None):
    from VideoSearch.models import Keyframe

    clip = entry.clip
    progress_str = f"[{progress[0]}/{progress[1]}] " if progress else ""
    
    if command:
        command.stdout.write(command.style_info(f"{progress_str}Processing clip {clip.id} (Video {clip.video.id}, frames {clip.start_frame}-{clip.end_frame})"))
    elif not pbar:  # Only print if not using progress bar
        print(f"{progress_str}[KeyframeExtraction] Processing clip {clip.id} (Video {clip.video.id}, frames {clip.start_frame}-{clip.end_frame})")

    Keyframe.objects.filter(clip=clip).delete()

    probs = entry.load_predictions()
    
    # Set pbar context for multipass function to suppress debug messages
    from VideoSearch.management.internal.extract_clips import multipass_predictions_to_scenes
    multipass_predictions_to_scenes._pbar = pbar
    
    change_regions = multipass_predictions_to_scenes(probs, 0.01, 1, 3, 1, 25, clip.fps())
    
    # Always show this critical info in progress bar or print
    clip_length = clip.end_frame - clip.start_frame + 1
    regions_msg = f"Detected {len(change_regions)} change regions"
    if pbar:
        pbar.set_postfix_str(f"Clip {clip.id} ({clip_length} frames): {regions_msg}")
        pbar.refresh()
    elif command:
        command.stdout.write(command.style_info(f"{regions_msg}."))
    else:
        print(f"[KeyframeExtraction] {regions_msg}.")

    for i, (start, end) in enumerate(change_regions, 1):
        potential_keyframe = int((start + end) / 2)
        
        # Calculate adaptive frames_to_compare based on region length
        region_length_frames = end - start + 1
        region_duration_seconds = region_length_frames / clip.fps()
        adaptive_frames_to_compare = calculate_frames_to_compare(region_duration_seconds)
        
        # Show progress through regions
        if pbar:
            pbar.set_postfix_str(f"Clip {clip.id} ({clip_length} frames): Processing region {i}/{len(change_regions)} ({region_length_frames}f, {adaptive_frames_to_compare}smp)")
            pbar.refresh()
        
        keyframes_before = Keyframe.objects.filter(clip=clip).count()
        
        try_for_potential_keyframe(
            feature_extractor,
            clip,
            potential_keyframe,
            start,
            end,
            int((end - start) * search_range_factor),
            threshold,
            adaptive_frames_to_compare
        )
        
        keyframes_after = Keyframe.objects.filter(clip=clip).count()
        keyframes_added = keyframes_after - keyframes_before
        
        # Show keyframes added for this region
        if pbar:
            pbar.set_postfix_str(f"Clip {clip.id} ({clip_length} frames): Region {i}/{len(change_regions)} → +{keyframes_added} keyframes")
            pbar.refresh()
        elif not command:
            print(f"[KeyframeExtraction] Region {i}/{len(change_regions)} → Added {keyframes_added} keyframes")

    keyframe_count = Keyframe.objects.filter(clip=clip).count()
    clip_length = clip.end_frame - clip.start_frame + 1
    result_msg = f"{progress_str}Extracted {keyframe_count} keyframes for clip {clip.id}"
    
    # Always show completion with tqdm.write when using progress bar
    if pbar:
        tqdm.write(f"[OK] Clip {clip.id} ({clip_length} frames): Extracted {keyframe_count} keyframes from {len(change_regions)} regions")
        pbar.set_postfix_str(f"Completed clip {clip.id}")
        pbar.refresh()
    elif command:
        command.stdout.write(command.style_success(result_msg))
    else:
        print(f"[KeyframeExtraction] {result_msg}")

    # GPU memory cleanup
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass  # torch not available
    
    entry.delete()
    return result_msg

def init_worker():
    """Initialize model only once per worker process."""
    import os
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ContentBasedVideoRetrieval.settings")
    import django
    django.setup()

    global feature_extractor
    from VideoSearch.utils.visual_feature_extractor import VisualFeatureExtractor
    feature_extractor = VisualFeatureExtractor(command=None)

def process_clip_entry_worker(entry_id, threshold, search_range_factor, frames_to_compare, progress=None, pbar=None):
    from VideoSearch.models import ClipPredictionCache
    import time

    global feature_extractor
    entry = ClipPredictionCache.objects.select_related("clip", "clip__video").get(id=entry_id)
    
    start_time = time.time()
    result = process_clip_entry(
        entry,
        feature_extractor,
        threshold,
        search_range_factor,
        frames_to_compare,
        progress,
        pbar,
        command=None
    )
    
    # Add timing info for slow clips (only when not using progress bar)
    elapsed = time.time() - start_time
    if elapsed > 10 and not pbar:  # Only log slow clips in non-progress mode
        print(f"[SLOW] Clip {entry.clip.id} took {elapsed:.1f}s")
    
    return result

def try_for_potential_keyframe(
    feature_extractor,
    clip,
    potential_keyframe: int,
    lower_bound: int,
    upper_bound: int,
    search_range: int,
    threshold: float,
    amount_of_frames_to_compare: int,
):
    """
    Tries to add new keyframes by sampling within a potential region of stability.
    Adds the most representative image if sufficiently different from existing keyframes.
    """
    start, end = compute_sampling_bounds(clip, potential_keyframe, lower_bound, upper_bound, search_range)
    num_frames = end - start + 1
    step_size = max(1, num_frames // amount_of_frames_to_compare)
    frame_offsets = [i for i in range(0, end - start + 1, step_size)]
    images = clip.get_selected_frame_images(frame_offsets)

    if not images:
        if feature_extractor.command:
            feature_extractor.command.stdout.write(feature_extractor.command.style_warning(f"No images found in range {start}-{end} for clip {clip.id}"))
        else:
            print(f"[KeyframeExtraction] No images found in range {start}-{end} for clip {clip.id}")
        return

    candidates = feature_extractor.get_candidates(images, start, step_size, clip, threshold)

    if(not candidates):
        return

    refine_and_store_keyframes(candidates, clip, feature_extractor, threshold)


def compute_sampling_bounds(clip, center_frame, lower_bound, upper_bound, search_range):
    search_range = abs(search_range)

    proposed_start = int(center_frame - search_range / 2)
    proposed_end = int(center_frame + search_range / 2)

    start = max(lower_bound, proposed_start)
    end = min(upper_bound, proposed_end)

    if end > clip.total_frames():
        end = clip.total_frames() - 1

    if start > end:
        return center_frame, center_frame+1
    else:
        return start, end


def refine_and_store_keyframes(candidates, clip, feature_extractor, threshold):
    from VideoSearch.models import Keyframe
    while candidates:
        best_frame = feature_extractor.select_representative(candidates)
        if not best_frame:
            break

        frame_number, features = best_frame
        Keyframe.create(
            clip,
            frame_number, 
            features["clip_emb"], 
            features["dino_emb"],
            features["histogram"],
            features["palette"],
            features["colorfulness"])

        candidates = [
            (frame, features)
            for frame, features in candidates
            if feature_extractor.distance_to_existing_keyframes(clip, features)[0] > threshold
        ]