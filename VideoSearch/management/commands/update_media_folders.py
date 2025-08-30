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
            '--no-process',
            action='store_true',
            help='Do NOT run processing pipeline (default: process videos)'
        )
        parser.add_argument(
            '--folder',
            type=str,
            help='Only scan specific folder by name'
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        process = not options['no_process']  # Default True, disable with --no-process
        folder_filter = options['folder']

        # Get active media folders
        folders = MediaFolderSetting.objects.filter(is_active=True)
        if folder_filter:
            folders = folders.filter(name=folder_filter)

        if not folders.exists():
            if folder_filter:
                self.stdout.write(self.style_error(f"ERROR: No active folder found with name: {folder_filter}"))
            else:
                self.stdout.write(self.style_error("ERROR: No active media folders configured"))
                self.stdout.write("TIP: Use 'python manage.py add_media_folder' to add some")
            return

        self.stdout.write(f">> Scanning {folders.count()} media folder(s)...")
        if dry_run:
            self.stdout.write(self.style_warning("*** DRY RUN MODE ***"))

        total_found = 0
        total_imported = 0
        total_skipped = 0

        # Collect all valid video files from all folders
        all_valid_paths = set()

        for folder in folders:
            self.stdout.write(f"\n[FOLDER] Processing: {folder.name}")
            self.stdout.write(f"   Path: {folder.path}")
            
            folder_path = Path(folder.path)
            if not folder.path_exists():
                self.stdout.write(self.style_error(f"   ERROR: Path no longer exists: {folder.path}"))
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
            self.stdout.write(f"   Videos found: {folder_count}")

            if dry_run:
                for video_path in sorted(video_files):
                    self.stdout.write(f"   -> Would scan: {video_path.name}")
            else:
                # Import videos from this folder
                imported, skipped = self.import_folder_videos(video_files)
                total_imported += imported
                total_skipped += skipped
                self.stdout.write(f"   Result: Imported {imported}, Skipped {skipped}")

        # Remove stale videos (only if not dry run)
        if not dry_run and all_valid_paths:
            self.remove_stale_videos(all_valid_paths)

        # Summary
        self.stdout.write(f"\n[SUMMARY]")
        self.stdout.write(f"   Total found: {total_found}")
        if not dry_run:
            self.stdout.write(f"   Imported: {total_imported}")
            self.stdout.write(f"   Skipped: {total_skipped}")

        # Run processing pipeline if requested
        if process and not dry_run:
            self.stdout.write(f"\n>> Running processing pipeline...")
            
            # Check what needs processing
            from VideoSearch.models import Clip, Keyframe
            videos_needing_processing = self.check_processing_status()
            
            if videos_needing_processing['total'] > 0:
                self.stdout.write(f"[PROCESSING STATUS]")
                self.stdout.write(f"   Videos without web proxies: {videos_needing_processing['no_web_proxy']}")
                self.stdout.write(f"   Videos without clips: {videos_needing_processing['no_clips']}")
                self.stdout.write(f"   Videos without keyframes: {videos_needing_processing['no_keyframes']}")
                self.stdout.write(f"   Videos without transcripts: {videos_needing_processing['no_transcripts']}")
                self.stdout.write(f"   Videos without objects: {videos_needing_processing['no_objects']}")
                
                # Run processing steps directly (skip import_videos since we just imported)
                from multiprocessing import cpu_count
                worker_clip = min(1, cpu_count())
                worker_keyframes = min(1, cpu_count())
                
                if videos_needing_processing['no_web_proxy'] > 0:
                    self.stdout.write(">> Creating Web Proxies...")
                    from VideoSearch.management.internal.create_web_videos import Command as CreateWebVideosCommand
                    cmd = CreateWebVideosCommand()
                    cmd.handle(max_height=480, quality=18)

                if videos_needing_processing['no_clips'] > 0:
                    self.stdout.write(">> Extracting Clips...")
                    from VideoSearch.management.internal.extract_clips import Command as ExtractClipsCommand
                    cmd = ExtractClipsCommand()
                    cmd.handle(workers=worker_clip)

                if videos_needing_processing['no_keyframes'] > 0:
                    self.stdout.write(">> Extracting Keyframes...")
                    import torch
                    from VideoSearch.management.internal.extract_keyframes import Command as ExtractKeyframesCommand
                    cmd = ExtractKeyframesCommand()
                    keyframe_kwargs = {
                        "search_range_factor": 0.95 if torch.cuda.is_available() else 0.5,
                        "frames_to_compare": 50 if torch.cuda.is_available() else 5,
                        "workers": worker_keyframes
                    }
                    cmd.handle(**keyframe_kwargs)

                if videos_needing_processing['no_transcripts'] > 0:
                    self.stdout.write(">> Extracting Audio Transcripts...")
                    from VideoSearch.management.internal.extract_audio_transcripts import Command as ExtractAudioTranscriptsCommand
                    cmd = ExtractAudioTranscriptsCommand()
                    cmd.handle(model_size="base", context_window=5.0)

                if videos_needing_processing['no_objects'] > 0:
                    self.stdout.write(">> Extracting Objects...")
                    from VideoSearch.management.internal.extract_objects import Command as ExtractObjectsCommand
                    cmd = ExtractObjectsCommand()
                    cmd.handle(batch_size=4)

                self.stdout.write("COMPLETE: Processing finished!")
            else:
                self.stdout.write("OK: All videos are already fully processed!")

        if dry_run:
            self.stdout.write(f"\nTIP: Run without --dry-run to actually import videos")
        elif not process:
            self.stdout.write(f"\nTIP: Processing was skipped (use without --no-process to enable)")

    def check_processing_status(self):
        """Check which videos need processing"""
        from VideoSearch.models import Clip, Keyframe
        from pathlib import Path
        
        all_videos = Video.objects.all()
        
        no_web_proxy = 0
        no_clips = 0  
        no_keyframes = 0
        no_transcripts = 0
        no_objects = 0
        
        for video in all_videos:
            # Check web proxy
            if not video.web_path or not Path(video.web_path).exists():
                no_web_proxy += 1
            
            # Check clips
            if not Clip.objects.filter(video=video).exists():
                no_clips += 1
            else:
                # Check keyframes (only for videos that have clips)
                clips_with_keyframes = Clip.objects.filter(
                    video=video,
                    keyframe__isnull=False
                ).distinct().count()
                total_clips = Clip.objects.filter(video=video).count()
                
                if clips_with_keyframes < total_clips:
                    no_keyframes += 1
                else:
                    # Check transcripts (only for videos with keyframes)
                    keyframes_with_transcripts = Keyframe.objects.filter(
                        clip__video=video,
                        transcript_embedding__isnull=False
                    ).count()
                    total_keyframes = Keyframe.objects.filter(clip__video=video).count()
                    
                    if keyframes_with_transcripts < total_keyframes:
                        no_transcripts += 1
                    else:
                        # Check objects (only for videos with transcripts)
                        keyframes_with_objects = Keyframe.objects.filter(
                            clip__video=video,
                            object_vector__isnull=False
                        ).count()
                        
                        if keyframes_with_objects < total_keyframes:
                            no_objects += 1
        
        return {
            'total': no_web_proxy + no_clips + no_keyframes + no_transcripts + no_objects,
            'no_web_proxy': no_web_proxy,
            'no_clips': no_clips,
            'no_keyframes': no_keyframes,
            'no_transcripts': no_transcripts,
            'no_objects': no_objects
        }

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
                self.stdout.write(self.style_error(f"   ERROR: Error importing {video_path.name}: {e}"))
                skipped += 1

        return imported, skipped

    def remove_stale_videos(self, valid_paths):
        """Remove videos from DB that no longer exist in any configured folder"""
        db_videos = Video.objects.all()
        removed_count = 0
        
        for video in db_videos:
            if str(Path(video.file_path).resolve()) not in valid_paths:
                self.stdout.write(f"   Removing stale video: {Path(video.file_path).name}")
                video.delete()  # Cascade deletion handles web proxies
                removed_count += 1
        
        if removed_count > 0:
            self.stdout.write(f"CLEANUP: Removed {removed_count} stale video(s)")

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