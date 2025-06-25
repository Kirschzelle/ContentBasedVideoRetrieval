from VideoSearch.management.base import StyledCommand as BaseCommand
from django.core.management import call_command

class Command(BaseCommand):
    help = "Run full import pipeline with default parameters."

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style_info("=== Importing Videos ==="))
        call_command("import_videos")

        self.stdout.write(self.style_info("=== Extracting Clips ==="))
        call_command("extract_clips")

        self.stdout.write(self.style_info("=== Extracting Keyframes ==="))
        call_command("extract_keyframes", debugging=True)

        self.stdout.write(self.style_success("Full import completed."))