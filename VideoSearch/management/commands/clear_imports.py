from VideoSearch.management.base import StyledCommand as BaseCommand
from VideoSearch.models import Video

class Command(BaseCommand):
    help = "Delete all imported videos and cascade clear related data."

    def handle(self, *args, **kwargs):
        count, _ = Video.objects.all().delete()
        if count > 0:
            self.stdout.write(self.style_success(f"Cleared {count} video(s) and related data."))
        else:
            self.stdout.write(self.style_error("No imported videos found. Nothing to clear."))