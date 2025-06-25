from VideoSearch.management.base import StyledCommand as BaseCommand
from django.core.management import call_command
import torch

class Command(BaseCommand):
    help = "Run full import pipeline with default parameters."

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style_info("=== Importing Videos ==="))
        call_command("import_videos")

        self.stdout.write(self.style_info("=== Extracting Clips ==="))
        call_command("extract_clips")

        self.stdout.write(self.style_info("=== Extracting Keyframes ==="))
        if torch.cuda.is_available():
            call_command("extract_keyframes", search_range_factor = 0.75, frames_to_compare = 10)
        else:
            call_command("extract_keyframes", search_range_factor = 0.5, frames_to_compare = 5)

        self.stdout.write(self.style_success("Full import completed."))