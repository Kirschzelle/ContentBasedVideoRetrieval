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
            
            if not folder.path:
                self.stdout.write(self.style_error(f"   ERROR: Path is empty for folder: {folder.name}"))
                continue
                
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

        if process and not dry_run:
            self.stdout.write(f"\n>> Running per-video processing pipeline...")
            
            from VideoSearch.models import Clip, Keyframe
            from multiprocessing import cpu_count
            
            all_videos = Video.objects.all()
            if not all_videos.exists():
                self.stdout.write("OK: No videos to process!")
                return
            
            worker_keyframes = min(1, cpu_count())
            processed_count = 0
            
            self.stdout.write(f"Processing {all_videos.count()} videos sequentially...")
            
            for video in all_videos:
                video_name = Path(video.file_path).name
                self.stdout.write(f"\n[VIDEO] {video_name} (ID: {video.id})")
                
                needs_web_proxy = not video.web_path or not Path(video.web_path).exists()
                if needs_web_proxy:
                    self.stdout.write("   >> Creating web proxy...")
                    if self.create_web_proxy_for_video(video):
                        video.refresh_from_db()
                    else:
                        self.stdout.write("   >> ERROR: Failed to create web proxy, skipping video")
                        continue
                else:
                    self.stdout.write("   >> Web proxy already exists")
                
                has_clips = Clip.objects.filter(video=video).exists()
                if not has_clips:
                    self.stdout.write("   >> Extracting clips...")
                    if self.extract_clips_for_video(video):
                        pass
                    else:
                        self.stdout.write("   >> ERROR: Failed to extract clips, skipping video")
                        continue
                else:
                    self.stdout.write("   >> Clips already exist")
                
                clips = Clip.objects.filter(video=video)
                if clips.exists():
                    keyframes_exist = Keyframe.objects.filter(clip__video=video).exists()
                    if not keyframes_exist:
                        self.stdout.write("   >> Processing clips (keyframes + audio + OCR + objects)...")
                        if self.process_clips_for_video(video):
                            pass
                        else:
                            self.stdout.write("   >> ERROR: Failed to process clips, skipping video")
                            continue
                    else:
                        self.stdout.write("   >> Keyframes already processed")
                
                processed_count += 1
                self.stdout.write(f"   >> COMPLETE: Video {processed_count}/{all_videos.count()}")
            
            self.stdout.write(f"\nCOMPLETE: Processed {processed_count} videos!")

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
            'no_clips': no_clips + no_web_proxy,
            'no_keyframes': no_keyframes + no_clips + no_web_proxy,
            'no_transcripts': no_transcripts + no_keyframes + no_clips + no_web_proxy,
            'no_objects': no_objects + no_transcripts + no_keyframes + no_clips + no_web_proxy
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

    def create_web_proxy_for_video(self, video):
        try:
            import subprocess
            input_path = Path(video.file_path)
            output_dir = Path('data/videos_web')
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{input_path.stem}.mp4"
            
            if output_path.exists():
                video.web_path = str(output_path)
                video.save()
                return True
                
            if not input_path.exists():
                self.stdout.write(f"   >> ERROR: Source file not found: {input_path}")
                return False
            
            cmd = [
                "ffmpeg", "-i", str(input_path),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-vf", "scale=-2:480:force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-profile:v", "main", "-level", "3.1", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "96k", "-ac", "2",
                "-movflags", "+faststart", "-avoid_negative_ts", "make_zero",
                "-y", str(output_path)
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            video.web_path = str(output_path)
            video.save()
            return True
            
        except Exception as e:
            self.stdout.write(f"   >> ERROR creating web proxy: {e}")
            return False
    
    def extract_clips_for_video(self, video):
        try:
            from VideoSearch.management.internal.extract_clips import process_video_for_clips
            result = process_video_for_clips(video.id, {}, None)
            self.stdout.write(f"   >> {result}")
            return True
        except Exception as e:
            self.stdout.write(f"   >> ERROR extracting clips: {e}")
            return False
    
    def process_clips_for_video(self, video):
        try:
            from VideoSearch.models import Clip
            from VideoSearch.utils.clip_processor import ClipProcessor
            
            clips = Clip.objects.filter(video=video)
            if not clips.exists():
                return True
            
            processor = ClipProcessor(whisper_model_size="base", command=self)
            
            for clip in clips:
                try:
                    keyframes = processor.process_clip(clip)
                    self.stdout.write(f"   >> Clip {clip.id}: {len(keyframes)} keyframes created")
                except Exception as e:
                    self.stdout.write(f"   >> ERROR processing clip {clip.id}: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self.stdout.write(f"   >> ERROR in clip processing: {e}")
            return False