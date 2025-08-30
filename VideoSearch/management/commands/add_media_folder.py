from VideoSearch.management.base import StyledCommand as BaseCommand
from VideoSearch.models import MediaFolderSetting
from pathlib import Path

class Command(BaseCommand):
    help = "Add a media folder to the system configuration."

    def add_arguments(self, parser):
        parser.add_argument(
            'name',
            type=str,
            help='Descriptive name for this media folder'
        )
        parser.add_argument(
            'path',
            type=str,
            help='Full path to the media folder'
        )
        parser.add_argument(
            '--no-recursive',
            action='store_true',
            help='Do NOT scan subdirectories (default: scan recursively)'
        )
        parser.add_argument(
            '--extensions',
            type=str,
            default='mp4,mov,mkv,avi,wmv,flv,webm,m4v',
            help='Comma-separated list of video extensions'
        )

    def handle(self, *args, **options):
        name = options['name']
        path = Path(options['path']).resolve()
        recursive = not options['no_recursive']  # Default True, disable with --no-recursive
        extensions = options['extensions']

        # Validate path
        if not path.exists():
            self.stdout.write(self.style_error(f"❌ Path does not exist: {path}"))
            return

        if not path.is_dir():
            self.stdout.write(self.style_error(f"❌ Path is not a directory: {path}"))
            return

        try:
            # Create or update media folder setting
            folder_setting, created = MediaFolderSetting.objects.update_or_create(
                name=name,
                defaults={
                    'path': str(path),
                    'recursive': recursive,
                    'extensions': extensions,
                    'is_active': True
                }
            )

            if created:
                self.stdout.write(self.style_success(f"✅ Added new media folder: {name}"))
            else:
                self.stdout.write(self.style_warning(f"⚠️  Updated existing media folder: {name}"))

            self.stdout.write(f"📁 Path: {path}")
            self.stdout.write(f"🔄 Recursive: {recursive}")
            self.stdout.write(f"📽️  Extensions: {extensions}")
            self.stdout.write(f"💡 Next: Run 'python manage.py update_media_folders' to scan for videos")

        except Exception as e:
            self.stdout.write(self.style_error(f"❌ Failed to add media folder: {e}"))