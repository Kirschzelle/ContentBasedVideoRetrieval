from django.core.management.base import BaseCommand
import colorama

colorama.init()  # auto-enables color output on Windows terminals

class StyledCommand(BaseCommand):
    def style_info(self, text):
        return f"{colorama.Fore.CYAN}{text}{colorama.Style.RESET_ALL}"

    def style_success(self, text):
        return f"{colorama.Fore.GREEN}{text}{colorama.Style.RESET_ALL}"

    def style_warning(self, text):
        return f"{colorama.Fore.YELLOW}{text}{colorama.Style.RESET_ALL}"

    def style_error(self, text):
        return f"{colorama.Fore.RED}{text}{colorama.Style.RESET_ALL}"