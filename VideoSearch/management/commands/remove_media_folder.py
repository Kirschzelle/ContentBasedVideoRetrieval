from VideoSearch.management.base import StyledCommand as BaseCommand
from VideoSearch.models import MediaFolderSetting, Video
from pathlib import Path

class Command(BaseCommand):
    help = "Remove a media folder from the system configuration."

    def add_arguments(self, parser):
        parser.add_argument(
            'name',
            type=str,
            help='Name of the media folder to remove'
        )
        parser.add_argument(
            '--keep-videos',
            action='store_true',
            help='Keep videos from this folder in the database (default: remove them)'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Skip confirmation prompt'
        )

    def handle(self, *args, **options):
        name = options['name']
        keep_videos = options['keep_videos']
        force = options['force']

        try:
            folder_setting = MediaFolderSetting.objects.get(name=name)
        except MediaFolderSetting.DoesNotExist:
            self.stdout.write(self.style_error(f"❌ Media folder '{name}' not found"))
            
            # Show available folders
            available = MediaFolderSetting.objects.all()
            if available.exists():
                self.stdout.write("Available folders:")
                for f in available:
                    self.stdout.write(f"  - {f.name}")
            else:
                self.stdout.write("No media folders configured")
            return

        # Count videos that would be affected
        folder_path = Path(folder_setting.path).resolve()
        affected_videos = []
        
        for video in Video.objects.all():
            video_path = Path(video.file_path).resolve()
            try:
                # Check if video is in this folder
                video_path.relative_to(folder_path)
                affected_videos.append(video)
            except ValueError:
                # Video is not in this folder
                continue

        # Display information
        self.stdout.write(f"📁 Media folder: {folder_setting.name}")
        self.stdout.write(f"   Path: {folder_setting.path}")
        self.stdout.write(f"   🎬 Videos in database from this folder: {len(affected_videos)}")

        if not keep_videos and affected_videos:
            self.stdout.write(self.style_warning(f"⚠️  {len(affected_videos)} videos will be REMOVED from database"))
            for video in affected_videos[:3]:  # Show first 3 as examples
                self.stdout.write(f"     - {video.file_name}")
            if len(affected_videos) > 3:
                self.stdout.write(f"     ... and {len(affected_videos) - 3} more")
        elif keep_videos and affected_videos:
            self.stdout.write(self.style_success(f"✅ {len(affected_videos)} videos will be KEPT in database"))

        # Confirmation
        if not force:
            action = "REMOVE FOLDER CONFIG ONLY" if keep_videos else "REMOVE FOLDER AND ALL ITS VIDEOS"
            confirm = input(f"\n{action} '{name}'? [y/N]: ")
            if confirm.lower() != 'y':
                self.stdout.write("Cancelled")
                return

        # Remove videos if not keeping them
        if not keep_videos and affected_videos:
            self.stdout.write(f"🗑️  Removing {len(affected_videos)} videos...")
            for video in affected_videos:
                video.delete()  # Cascade deletion handles web proxies, clips, keyframes
            self.stdout.write(f"✅ Removed {len(affected_videos)} videos and their data")

        # Remove folder setting
        folder_setting.delete()
        self.stdout.write(self.style_success(f"✅ Removed media folder configuration: {name}"))

        # Summary
        if keep_videos and affected_videos:
            self.stdout.write(f"💡 {len(affected_videos)} videos from this folder remain in the database")
            self.stdout.write("   They will be marked as 'stale' in future scans if the folder path is gone")
        
        remaining_folders = MediaFolderSetting.objects.count()
        if remaining_folders == 0:
            self.stdout.write("📂 No media folders remain configured")
        else:
            self.stdout.write(f"📂 {remaining_folders} media folder(s) remaining")