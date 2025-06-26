from tkinter import CENTER
from VideoSearch.management.base import StyledCommand as BaseCommand
from VideoSearch.management.commands.extract_clips import multipass_predictions_to_scenes
from VideoSearch.models import ClipPredictionCache, Clip, Keyframe
from VideoSearch.utils.visual_feature_extractor import VisualFeatureExtractor
import time

class Command(BaseCommand):
    help = "Extract keyframes from newly extracted clips."

    def add_arguments(self, parser):
        parser.add_argument('--threshold', type=float, default=0.35, help='Distance threshold for keyframe uniqueness.')
        parser.add_argument('--search-range-factor', type=float, default=0.95, help='Fraction of region used to search around potential keyframe.')
        parser.add_argument('--frames-to-compare', type=int, default=25, help='How many frames to sample when searching for keyframes.')

    def handle(self, *args, **kwargs):
        threshold = kwargs.get('threshold', 0.35)
        search_range_factor = kwargs.get('search_range_factor', 0.95)
        frames_to_compare = kwargs.get('frames_to_compare', 25)

        candidates = ClipPredictionCache.objects.select_related("clip").all()
        if not candidates:
            self.stdout.write(self.style_warning("No clips require keyframe extraction."))
            return

        feature_extractor = VisualFeatureExtractor(command=self)

        for entry in candidates:
            try:
                process_clip_entry(
                    entry,
                    feature_extractor,
                    threshold,
                    search_range_factor,
                    frames_to_compare,
                    self
                )
            except Exception as e:
                self.stdout.write(self.style_error(f"Exception during keyframe extraction: {e}"))

def process_clip_entry(entry, feature_extractor: VisualFeatureExtractor, threshold, search_range_factor, frames_to_compare, command=None):
    clip = entry.clip
    if command:
        command.stdout.write(command.style_info(f"Processing clip {clip.id} (Video {clip.video.id}, frames {clip.start_frame}-{clip.end_frame})"))

    Keyframe.objects.filter(clip=clip).delete()

    probs = entry.load_predictions()
    change_regions = multipass_predictions_to_scenes(probs, 0.01, 1, 3, 1, 25, clip.fps())
    if command:
        command.stdout.write(command.style_info(f"Detected {len(change_regions)} change regions."))

    for start, end in change_regions:
        potential_keyframe = int((start + end) / 2)
        try_for_potential_keyframe(
            feature_extractor,
            clip,
            potential_keyframe,
            start,
            end,
            int((end - start) * search_range_factor),
            threshold,
            frames_to_compare
        )

    if command:
        command.stdout.write(command.style_success(f"Extracted {Keyframe.objects.filter(clip=clip).count()} keyframes."))

    entry.delete()

def try_for_potential_keyframe(
    feature_extractor: VisualFeatureExtractor,
    clip: Clip,
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
    t000 = time.perf_counter()
    start, end = compute_sampling_bounds(clip, potential_keyframe, lower_bound, upper_bound, search_range)
    t00 = time.perf_counter()
    num_frames = end - start + 1
    step_size = max(1, num_frames // amount_of_frames_to_compare)
    frame_offsets = [i for i in range(0, end - start + 1, step_size)]
    images = clip.get_selected_frame_images(frame_offsets)

    if not images:
        if feature_extractor.command:
            feature_extractor.command.stdout.write(feature_extractor.command.style_warning(f"No images found in range {start}-{end} for clip {clip.id}"))
        return

    t0 = time.perf_counter()

    candidates = feature_extractor.get_candidates(images, start, step_size, clip, threshold)

    t1 = time.perf_counter()

    if(not candidates):
        return

    refine_and_store_keyframes(candidates, clip, feature_extractor, threshold)

    t2 = time.perf_counter()

    if feature_extractor.command:
        feature_extractor.command.stdout.write(feature_extractor.command.style_info(
            f"Sampling Bounds: {(t00 - t000) * 1000:.1f}ms | Image loading: {(t0 - t00) * 1000:.1f}ms | Keyframe candidate search: {(t1 - t0) * 1000:.1f}ms | refine/store: {(t2 - t1) * 1000:.1f}ms"
        ))


def compute_sampling_bounds(clip : Clip, center_frame, lower_bound, upper_bound, search_range):
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
    while candidates:
        t0 = time.perf_counter()
        best_frame = feature_extractor.select_representative(candidates)
        t1 = time.perf_counter()

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
        t2 = time.perf_counter()

        candidates = [
            (frame, features)
            for frame, features in candidates
            if feature_extractor.distance_to_existing_keyframes(clip, features)[0] > threshold
        ]
        t3 = time.perf_counter()

        if feature_extractor.command:
            feature_extractor.command.stdout.write(feature_extractor.command.style_info(
                f"Select: {(t1 - t0) * 1000:.1f}ms | Store: {(t2 - t1) * 1000:.1f}ms | Filter: {(t3 - t2) * 1000:.1f}ms"
            ))