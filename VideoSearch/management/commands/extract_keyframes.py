from tkinter import CENTER
from VideoSearch.management.base import StyledCommand as BaseCommand
from VideoSearch.management.commands.extract_clips import multipass_predictions_to_scenes
from VideoSearch.models import ClipPredictionCache, Clip, Keyframe
from VideoSearch.utils.embeddings import ImageEmbedder
import numpy as np

class Command(BaseCommand):
    help = "Extract keyframes from newly extracted clips."


    def handle(self, *args, **kwargs):
        candidates = ClipPredictionCache.objects.select_related("clip").all()

        if not candidates:
            self.stdout.write(self.style_warning(f"No clips require keyframe extraction."))
            return

        image_embedder = ImageEmbedder(command=self)

        for entry in candidates:
            clip = entry.clip
            self.stdout.write(self.style_info(f"Processing clip {clip.id} (Video {clip.video.id}, frames {clip.start_frame}-{clip.end_frame})"))

            deleted_count, _ = Keyframe.objects.filter(clip=clip).delete()
            if deleted_count > 0:
                self.stdout.write(self.style_warning(f"Deleted {deleted_count} old keyframes."))

            probs = entry.load_predictions()
            change_regions = multipass_predictions_to_scenes(probs, 0.01, 1, 5, 2, 50, clip.fps())
            self.stdout.write(self.style_info(f"Detected {len(change_regions)} change regions."))

            total_added = 0
            for start, end in change_regions:
                potential_keyframe = int((start + end) / 2)
                added = try_for_potential_keyframe(
                    image_embedder,
                    clip,
                    potential_keyframe,
                    start,
                    end,
                    int((end - start) * 0.95),
                    0.35,
                    25
                )
                if added:
                    total_added += 1

            self.stdout.write(self.style_success(f"Extracted {Keyframe.objects.filter(clip=clip).count()} keyframes."))

            entry.delete()

        # For debugging
        from pathlib import Path

        DEBUG_DIR = Path("data/debug")
        import shutil

        if DEBUG_DIR.exists():
            shutil.rmtree(DEBUG_DIR)
        DEBUG_DIR.mkdir(parents=True)

        for keyframe in Keyframe.objects.all():
            img = keyframe.clip.get_frame_image(keyframe.frame)
            if img is not None:
                filename = f"video{keyframe.clip.video.id}_clip{keyframe.clip.id}_frame{keyframe.frame}.jpg"
                img.save(DEBUG_DIR / filename)


def try_for_potential_keyframe(
    embedder: ImageEmbedder,
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
    start, end = compute_sampling_bounds(clip, potential_keyframe, lower_bound, upper_bound, search_range)
    images = clip.get_frame_range_images(start, end)

    if not images:
        return

    num_frames = end - start + 1
    step_size = max(1, num_frames // amount_of_frames_to_compare)

    candidates = collect_embedding_candidates(images, start, step_size, clip, embedder, threshold)

    while(not candidates):
        return

    refine_and_store_keyframes(candidates, clip, embedder, threshold)


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


def collect_embedding_candidates(images, start, step_size, clip, embedder, threshold):
    candidates = []

    for i, image in enumerate(images):
        if i != 0 and i % step_size != 0:
            continue

        frame_number = start + i
        clip_emb, dino_emb = embedder.get_combined_embedding(image)

        min_dist, _ = embedder.get_distance_to_existing_keyframes(clip, clip_emb, dino_emb)
        if min_dist > threshold:
            candidates.append((frame_number, clip_emb, dino_emb))

    return candidates


def refine_and_store_keyframes(candidates, clip, embedder, threshold):
    while candidates:
        best_frame = select_representative_keyframe(candidates, embedder)

        if not best_frame:
            break

        frame_number, emb_clip, emb_dino = best_frame
        Keyframe.create(clip, frame_number, emb_clip, emb_dino)

        candidates = [
            (frame, c_emb, d_emb)
            for frame, c_emb, d_emb in candidates
            if embedder.get_distance_to_existing_keyframes(clip, c_emb, d_emb)[0] > threshold
        ]

def select_representative_keyframe(candidates, embedder):
    """
    Selects the most representative keyframe from a list of (frame, clip_emb, dino_emb) tuples,
    based on lowest average distance to other candidates.

    :param candidates: List of tuples (frame_number, clip_embedding, dino_embedding)
    :param embedder: ImageEmbedder instance
    :return: Tuple (frame_number, clip_embedding, dino_embedding)
    """
    best_frame = None
    best_score = float("inf")

    for i, (frame_i, emb_clip_i, emb_dino_i) in enumerate(candidates):
        similarities = [
            embedder.calculate_combined_distance(emb_clip_i, emb_dino_i, emb_clip_j, emb_dino_j)
            for j, (_, emb_clip_j, emb_dino_j) in enumerate(candidates)
            if i != j
        ]
        median_similarity = np.median(similarities)

        if median_similarity < best_score:
            best_score = median_similarity
            best_frame = (frame_i, emb_clip_i, emb_dino_i)

    return best_frame
