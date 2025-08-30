from VideoSearch.management.base import StyledCommand as BaseCommand
from VideoSearch.models import Video
from pathlib import Path
import subprocess
import os

class Command(BaseCommand):
    pass

    def add_arguments(self, parser):
        parser.add_argument(
            '--output-dir',
            type=str,
            default='data/videos_web',
            help='Directory to store web videos (default: data/videos_web)'
        )
        parser.add_argument(
            '--max-height',
            type=int,
            default=480,
            help='Maximum height for web videos (default: 480p)'
        )
        parser.add_argument(
            '--quality',
            type=int,
            default=18,
            help='Video quality CRF value (18-32, lower=better quality, default: 18)'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Overwrite existing web videos'
        )
        parser.add_argument(
            '--replace-originals',
            action='store_true',
            help='Replace original videos with web versions (saves storage)'
        )

    def handle(self, *args, **options):
        output_dir = Path(options['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        max_height = options['max_height']
        quality = options['quality']
        force = options['force']
        replace_originals = options['replace_originals']

        videos = Video.objects.all()
        self.stdout.write(f"Creating web videos for {len(videos)} videos...")

        if replace_originals:
            self.stdout.write(
                self.style_warning("⚠️  --replace-originals will delete original files after conversion!")
            )
            confirm = input("Continue? [y/N]: ")
            if confirm.lower() != 'y':
                return

        total_size_before = 0
        total_size_after = 0

        for video in videos:
            input_path = Path(video.file_path)
            
            if replace_originals:
                # Replace in same location with .mp4 extension
                output_path = input_path.with_suffix('.mp4')
            else:
                # Create web proxy in separate directory
                output_path = output_dir / f"{input_path.stem}.mp4"
            
            if output_path.exists() and not force:
                self.stdout.write(f"SKIP: {input_path.name} (already exists)")
                continue

            if not input_path.exists():
                self.stdout.write(self.style_error(f"ERROR: Source file not found: {input_path}"))
                continue

            self.stdout.write(f">> Processing {input_path.name}...")
            
            # Get input file size
            input_size = input_path.stat().st_size
            total_size_before += input_size
            
            # FFmpeg command for small, web-optimized files
            cmd = [
                "ffmpeg",
                "-i", str(input_path),
                
                # Video settings - optimized for small file size
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", str(quality),
                "-vf", f"scale=-2:{max_height}:force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2",  # Scale down and pad to even dimensions
                "-profile:v", "main",
                "-level", "3.1",
                "-pix_fmt", "yuv420p",
                
                # Audio settings - compressed
                "-c:a", "aac",
                "-b:a", "96k",
                "-ac", "2",
                
                # Web optimization
                "-movflags", "+faststart",
                "-avoid_negative_ts", "make_zero",
                
                # Overwrite policy
                "-y" if force else "-n",
                str(output_path)
            ]
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                # Get output file size
                output_size = output_path.stat().st_size
                total_size_after += output_size
                
                input_mb = input_size / (1024 * 1024)
                output_mb = output_size / (1024 * 1024)
                compression_ratio = (1 - output_size/input_size) * 100
                
                self.stdout.write(
                    self.style_success(
                        f"DONE: {input_path.name} -> {output_path.name} "
                        f"({input_mb:.1f}MB → {output_mb:.1f}MB, {compression_ratio:.1f}% smaller)"
                    )
                )
                
                # Update the video model
                if replace_originals:
                    # Delete original and update file_path
                    if input_path != output_path:  # Don't delete if same file
                        input_path.unlink()
                    video.file_path = str(output_path)
                else:
                    # Set web_path for proxy
                    video.web_path = str(output_path)
                
                video.save()
                
            except subprocess.CalledProcessError as e:
                self.stdout.write(
                    self.style_error(f"❌ Failed to convert {input_path.name}")
                )
                if e.stderr:
                    self.stdout.write(f"   FFmpeg error: {e.stderr}")
            except Exception as e:
                self.stdout.write(
                    self.style_error(f"❌ Error processing {input_path.name}: {e}")
                )

        # Summary
        total_before_gb = total_size_before / (1024**3)
        total_after_gb = total_size_after / (1024**3)
        total_saved = (1 - total_size_after/total_size_before) * 100 if total_size_before > 0 else 0
        
        self.stdout.write(self.style_success("\nCOMPLETE: Conversion finished!"))
        self.stdout.write(f"STATS: Total size: {total_before_gb:.2f}GB -> {total_after_gb:.2f}GB ({total_saved:.1f}% saved)")
        
        if replace_originals:
            self.stdout.write("INFO: Original files replaced with web-optimized versions")
        else:
            self.stdout.write(f"OUTPUT: Web videos saved to: {output_dir.resolve()}")
            
        self.stdout.write("SUCCESS: Videos are now web-compatible and will play in browsers!")