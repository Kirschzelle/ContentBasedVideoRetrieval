from VideoSearch.management.base import StyledCommand as BaseCommand
from django.core.management import call_command
import torch

class Command(BaseCommand):
    help = "Run full import pipeline with default parameters."

    def add_arguments(self, parser):
        parser.add_argument('--workers_clip', type=int, default=None, help="Number of multiprocessing workers to use for clip/keyframe extraction.")
        parser.add_argument('--workers_keyframes', type=int, default=None, help="Number of multiprocessing workers to use for clip/keyframe extraction.")

    def handle(self, *args, **kwargs):
        from multiprocessing import cpu_count

        worker_clip = kwargs.get("workers_clip")
        worker_keyframes = kwargs.get("workers_keyframes")
        worker_clip = worker_clip if worker_clip is not None else min(1, cpu_count())
        worker_keyframes = worker_keyframes if worker_keyframes is not None else min(1, cpu_count())

        self.stdout.write(self.style_info("=== Importing Videos ==="))
        call_command("import_videos")

        self.stdout.write(self.style_info("=== Extracting Clips ==="))
        call_command("extract_clips", workers=worker_clip)

        self.stdout.write(self.style_info("=== Extracting Keyframes ==="))
        keyframe_kwargs = {
            "search_range_factor": 0.95 if torch.cuda.is_available() else 0.5,
            "frames_to_compare": 50 if torch.cuda.is_available() else 5,
            "workers": worker_keyframes
        }
        call_command("extract_keyframes", **keyframe_kwargs)

        self.stdout.write(self.style_success("Full import completed."))