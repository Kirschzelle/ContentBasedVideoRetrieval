from django.core.management.base import BaseCommand

class StyledCommand(BaseCommand):
    def style_info(self, text):
        return f"\033[36m{text}\033[0m"