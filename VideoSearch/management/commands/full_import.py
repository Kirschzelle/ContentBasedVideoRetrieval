from VideoSearch.management.base import StyledCommand as BaseCommand
from django.core.management import call_command
import torch

class Command(BaseCommand):
    help = "Run full import pipeline with default parameters."

    def add_arguments(self, parser):
        parser.add_argument('--workers_clip', type=int, default=None, help="Number of multiprocessing workers to use for clip extraction.")
        parser.add_argument('--workers_keyframes', type=int, default=None, help="Number of multiprocessing workers to use for keyframe processing.")
        parser.add_argument('--whisper_model', type=str, default='base', choices=['tiny', 'base', 'small', 'medium', 'large'], help="Whisper model size for audio transcription.")

    def handle(self, *args, **kwargs):
        from multiprocessing import cpu_count

        worker_clip = kwargs.get("workers_clip")
        worker_keyframes = kwargs.get("workers_keyframes")
        whisper_model = kwargs.get("whisper_model", "base")
        worker_clip = worker_clip if worker_clip is not None else min(1, cpu_count())
        worker_keyframes = worker_keyframes if worker_keyframes is not None else min(1, cpu_count())

        self.stdout.write(self.style_info("=== Importing Videos ==="))
        call_command("import_videos")

        self.stdout.write(self.style_info("=== Creating Web Proxies ==="))
        call_command("create_web_videos", max_height=480, quality=18)

        self.stdout.write(self.style_info("=== Extracting Clips ==="))
        call_command("extract_clips", workers=worker_clip)

        self.stdout.write(self.style_info("=== Processing Clips (Audio + Visual Features) ==="))
        process_kwargs = {
            "whisper_model": whisper_model,
            "workers": worker_keyframes
        }
        call_command("process_clips_v2", **process_kwargs)

        self.stdout.write(self.style_success("Full import completed."))