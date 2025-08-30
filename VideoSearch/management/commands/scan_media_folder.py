from VideoSearch.management.base import StyledCommand as BaseCommand
from VideoSearch.models import Video
from pathlib import Path
import subprocess
import time

class Command(BaseCommand):
    help = "Scan any media folder and import video metadata into the database."

    def add_arguments(self, parser):
        parser.add_argument(
            'media_path',
            type=str,
            help='Path to the media folder to scan (can be anywhere on the system)'
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
            help='Comma-separated list of video extensions (default: mp4,mov,mkv,avi,wmv,flv,webm,m4v)'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be imported without actually importing'
        )

    def handle(self, *args, **options):
        media_path = Path(options['media_path']).resolve()
        recursive = not options['no_recursive']  # Default True, disable with --no-recursive
        extensions = [ext.strip().lower() for ext in options['extensions'].split(',')]
        dry_run = options['dry_run']

        if not media_path.exists():
            self.stdout.write(self.style_error(f"❌ Media path does not exist: {media_path}"))
            return

        if not media_path.is_dir():
            self.stdout.write(self.style_error(f"❌ Path is not a directory: {media_path}"))
            return

        self.stdout.write(f"📁 Scanning media folder: {media_path}")
        self.stdout.write(f"🔍 Extensions: {', '.join(extensions)}")
        self.stdout.write(f"📂 Recursive: {'Yes' if recursive else 'No'}")
        
        if dry_run:
            self.stdout.write(self.style_warning("🧪 DRY RUN MODE - No files will be imported"))

        # Find video files
        video_files = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in media_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower().lstrip('.') in extensions:
                video_files.append(file_path)

        if not video_files:
            self.stdout.write(self.style_warning(f"⚠️  No video files found in {media_path}"))
            return

        self.stdout.write(f"🎬 Found {len(video_files)} video files")
        
        # Remove stale videos (videos in DB that no longer exist)
        if not dry_run:
            valid_paths = set(str(f.resolve()) for f in video_files)
            self.remove_stale_videos(valid_paths)

        # Process each video file
        imported_count = 0
        skipped_count = 0

        for video_path in sorted(video_files):
            try:
                if dry_run:
                    self.stdout.write(f"📽️  Would import: {video_path.name}")
                    imported_count += 1
                else:
                    result = self.import_video_file(video_path)
                    if "imported" in result.lower():
                        imported_count += 1
                    else:
                        skipped_count += 1
                    self.stdout.write(result)
                    
            except Exception as e:
                self.stdout.write(self.style_error(f"❌ Error processing {video_path.name}: {e}"))

        # Summary
        if dry_run:
            self.stdout.write(self.style_success(f"🧪 DRY RUN: Would import {imported_count} videos"))
        else:
            self.stdout.write(self.style_success(
                f"✅ Import completed: {imported_count} imported, {skipped_count} skipped"
            ))
            self.stdout.write(f"💡 Next step: Run 'python manage.py full_import' to process the imported videos")

    def remove_stale_videos(self, valid_paths):
        """Remove videos from DB that no longer exist on disk"""
        db_videos = list(Video.objects.all())
        removed_count = 0
        
        for video in db_videos:
            if str(Path(video.file_path).resolve()) not in valid_paths:
                self.stdout.write(f"🗑️  Removing stale video: {Path(video.file_path).name}")
                video.delete()  # This will also delete web proxy via cascade
                removed_count += 1
        
        if removed_count > 0:
            self.stdout.write(f"🧹 Removed {removed_count} stale videos")

    def import_video_file(self, video_path):
        """Import a single video file, similar to import_videos logic"""
        video_path_str = str(video_path.resolve())
        
        # Check if already imported
        if Video.objects.filter(file_path=video_path_str).exists():
            return f"⏩ Skipping {video_path.name} - already imported"

        # Get video metadata using ffprobe
        try:
            metadata = self.get_video_metadata(video_path)
            if not metadata:
                return f"❌ Failed to get metadata for {video_path.name}"
            
            # Create video record
            video = Video.objects.create(
                file_path=video_path_str,
                frame_count=metadata['frame_count'],
                fps_num=metadata['fps_num'], 
                fps_den=metadata['fps_den'],
                resolution=metadata['resolution']
            )
            
            return f"✅ Imported {video_path.name} ({metadata['resolution']}, {metadata['frame_count']} frames)"
            
        except Exception as e:
            return f"❌ Failed to import {video_path.name}: {e}"

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
            import json
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
                'frame_count': max(frame_count, 1),  # Ensure at least 1 frame
                'fps_num': fps_num,
                'fps_den': fps_den,
                'resolution': f"{width}x{height}"
            }
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            return None