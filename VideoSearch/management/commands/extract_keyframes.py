from VideoSearch.management.base import StyledCommand as BaseCommand
from VideoSearch.management.commands.extract_clips import multipass_predictions_to_scenes
from VideoSearch.models import ClipPredictionCache, Clip
from VideoSearch.utils.embeddings import ImageEmbedder

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
            probs = entry.load_predictions()
       
            change_regions = multipass_predictions_to_scenes(probs, 0.2, 0.5, 5, 2, 50, clip.fps())

            keyframes = []

            for start, end in change_regions:
                potential_keyframe = (start + end) / 2
                try_for_potential_keyframe(image_embedder, clip, keyframes, potential_keyframe, start, end, int(clip.duration(end-start) * 0.25))

            for keyframe in keyframes:
                # Add keyframe to models (still need to define it - skip this for now)
                pass

            entry.delete()

def try_for_potential_keyframe(embedder, clip, keyframes, potential_keyframe, lower_bound, upper_bound, search_range):
    """
    Adds a new keyframe to the list if it's significantly different from existing keyframes.
    Optionally refines the selection within a search window.
    """
    pass
