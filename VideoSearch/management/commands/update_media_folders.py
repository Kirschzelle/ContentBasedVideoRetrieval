from VideoSearch.management.base import StyledCommand as BaseCommand
from VideoSearch.models import MediaFolderSetting, Video
from django.core.management import call_command
from pathlib import Path
import subprocess
import json

class Command(BaseCommand):
    help = "Scan all configured media folders and update video database."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be scanned without importing'
        )
        parser.add_argument(
            '--process',
            action='store_true',
            help='Run full processing pipeline after scanning'
        )
        parser.add_argument(
            '--folder',
            type=str,
            help='Only scan specific folder by name'
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        process = options['process']
        folder_filter = options['folder']

        # Get active media folders
        folders = MediaFolderSetting.objects.filter(is_active=True)
        if folder_filter:
            folders = folders.filter(name=folder_filter)

        if not folders.exists():
            if folder_filter:
                self.stdout.write(self.style_error(f"❌ No active folder found with name: {folder_filter}"))
            else:
                self.stdout.write(self.style_error("❌ No active media folders configured"))
                self.stdout.write("💡 Use 'python manage.py add_media_folder' to add some")
            return

        self.stdout.write(f"🔍 Scanning {folders.count()} media folder(s)...")
        if dry_run:
            self.stdout.write(self.style_warning("🧪 DRY RUN MODE"))

        total_found = 0
        total_imported = 0
        total_skipped = 0

        # Collect all valid video files from all folders
        all_valid_paths = set()

        for folder in folders:
            self.stdout.write(f"\n📁 Processing: {folder.name}")
            self.stdout.write(f"   Path: {folder.path}")
            
            folder_path = Path(folder.path)
            if not folder.path_exists():
                self.stdout.write(self.style_error(f"   ❌ Path no longer exists: {folder.path}"))
                continue

            # Find video files in this folder
            pattern = "**/*" if folder.recursive else "*"
            video_files = []
            
            for file_path in folder_path.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower().lstrip('.') in folder.extension_list:
                    video_files.append(file_path)
                    all_valid_paths.add(str(file_path.resolve()))

            folder_count = len(video_files)
            total_found += folder_count
            self.stdout.write(f"   🎬 Found: {folder_count} videos")

            if dry_run:
                for video_path in sorted(video_files):
                    self.stdout.write(f"   📽️  Would scan: {video_path.name}")
            else:
                # Import videos from this folder
                imported, skipped = self.import_folder_videos(video_files)
                total_imported += imported
                total_skipped += skipped
                self.stdout.write(f"   ✅ Imported: {imported}, Skipped: {skipped}")

        # Remove stale videos (only if not dry run)
        if not dry_run and all_valid_paths:
            self.remove_stale_videos(all_valid_paths)

        # Summary
        self.stdout.write(f"\n📊 Summary:")
        self.stdout.write(f"   🎬 Total found: {total_found}")
        if not dry_run:
            self.stdout.write(f"   ✅ Imported: {total_imported}")
            self.stdout.write(f"   ⏩ Skipped: {total_skipped}")

        # Run processing pipeline if requested
        if process and not dry_run:
            # Get total videos (imported + existing) to check if we have anything to process
            total_videos = Video.objects.count()
            if total_videos > 0:
                self.stdout.write(f"\n🚀 Running processing pipeline for {total_videos} videos...")
                
                # Run processing steps directly (skip import_videos since we just imported)
                from multiprocessing import cpu_count
                worker_clip = min(1, cpu_count())
                worker_keyframes = min(1, cpu_count())
                
                self.stdout.write("=== Creating Web Proxies ===")
                call_command("create_web_videos", max_height=480, quality=18)

                self.stdout.write("=== Extracting Clips ===")
                call_command("extract_clips", workers=worker_clip)

                self.stdout.write("=== Extracting Keyframes ===")
                import torch
                keyframe_kwargs = {
                    "search_range_factor": 0.95 if torch.cuda.is_available() else 0.5,
                    "frames_to_compare": 50 if torch.cuda.is_available() else 5,
                    "workers": worker_keyframes
                }
                call_command("extract_keyframes", **keyframe_kwargs)

                self.stdout.write("=== Extracting Objects ===")
                call_command("extract_objects", batch_size=4)

                self.stdout.write("🎉 Complete processing finished!")
            else:
                self.stdout.write("⚠️  No videos to process")

        if dry_run:
            self.stdout.write(f"\n💡 Run without --dry-run to actually import videos")
        elif total_imported > 0 and not process:
            self.stdout.write(f"\n💡 Run with --process to automatically process imported videos")

    def import_folder_videos(self, video_files):
        """Import videos from a list of file paths"""
        imported = 0
        skipped = 0

        for video_path in video_files:
            try:
                video_path_str = str(video_path.resolve())
                
                # Check if already imported
                if Video.objects.filter(file_path=video_path_str).exists():
                    skipped += 1
                    continue

                # Get metadata and create video record
                metadata = self.get_video_metadata(video_path)
                if metadata:
                    Video.objects.create(
                        file_path=video_path_str,
                        frame_count=metadata['frame_count'],
                        fps_num=metadata['fps_num'], 
                        fps_den=metadata['fps_den'],
                        resolution=metadata['resolution']
                    )
                    imported += 1
                else:
                    skipped += 1

            except Exception as e:
                self.stdout.write(self.style_error(f"   ❌ Error importing {video_path.name}: {e}"))
                skipped += 1

        return imported, skipped

    def remove_stale_videos(self, valid_paths):
        """Remove videos from DB that no longer exist in any configured folder"""
        db_videos = Video.objects.all()
        removed_count = 0
        
        for video in db_videos:
            if str(Path(video.file_path).resolve()) not in valid_paths:
                self.stdout.write(f"🗑️  Removing stale video: {Path(video.file_path).name}")
                video.delete()  # Cascade deletion handles web proxies
                removed_count += 1
        
        if removed_count > 0:
            self.stdout.write(f"🧹 Removed {removed_count} stale video(s)")

    def get_video_metadata(self, video_path):
        """Extract video metadata using ffprobe"""
        cmd = [
            "ffprobe", 
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams", 
            "-show_format",
            str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                return None
            
            # Extract metadata
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            
            # Parse frame rate
            r_frame_rate = video_stream.get('r_frame_rate', '30/1')
            fps_parts = r_frame_rate.split('/')
            fps_num = int(fps_parts[0])
            fps_den = int(fps_parts[1]) if len(fps_parts) > 1 else 1
            
            # Get frame count
            frame_count = int(video_stream.get('nb_frames', 0))
            if frame_count == 0:
                # Fallback: estimate from duration and fps
                duration = float(data['format'].get('duration', 0))
                if duration > 0:
                    frame_count = int(duration * fps_num / fps_den)
            
            return {
                'frame_count': max(frame_count, 1),
                'fps_num': fps_num,
                'fps_den': fps_den,
                'resolution': f"{width}x{height}"
            }
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
            return None