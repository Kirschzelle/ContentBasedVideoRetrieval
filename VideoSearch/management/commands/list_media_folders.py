from VideoSearch.management.base import StyledCommand as BaseCommand
from VideoSearch.models import MediaFolderSetting
from pathlib import Path

class Command(BaseCommand):
    help = "List all configured media folders."

    def add_arguments(self, parser):
        parser.add_argument(
            '--check-paths',
            action='store_true',
            help='Check if folder paths still exist'
        )

    def handle(self, *args, **options):
        check_paths = options['check_paths']
        
        folders = MediaFolderSetting.objects.all()
        
        if not folders.exists():
            self.stdout.write(self.style_warning("No media folders configured yet"))
            self.stdout.write("TIP: Use 'python manage.py add_media_folder' to add some")
            return

        self.stdout.write(f"Configured Media Folders ({folders.count()}):")
        self.stdout.write("-" * 50)

        for folder in folders:
            status_icon = "ACTIVE" if folder.is_active else "INACTIVE"
            self.stdout.write(f"[{status_icon}] {folder.name}")
            self.stdout.write(f"   Path: {folder.path}")
            self.stdout.write(f"   Recursive: {folder.recursive}")
            self.stdout.write(f"   Extensions: {folder.extensions}")
            self.stdout.write(f"   Active: {folder.is_active}")
            
            if check_paths:
                exists = folder.path_exists()
                path_status = "exists" if exists else "missing"
                self.stdout.write(f"   Status: {path_status}")
            
            self.stdout.write(f"   Added: {folder.created_at.strftime('%Y-%m-%d %H:%M')}")
            self.stdout.write("")

        active_count = folders.filter(is_active=True).count()
        self.stdout.write(f"SUMMARY: {active_count} active, {folders.count() - active_count} inactive")
        
        if active_count > 0:
            self.stdout.write("TIP: Run 'python manage.py update_media_folders' to scan for videos")