from VideoSearch.management.base import StyledCommand as BaseCommand
from django.core.management import call_command
import torch

class Command(BaseCommand):
    help = "Complete import workflow from any media folder."

    def add_arguments(self, parser):
        parser.add_argument(
            'media_path',
            type=str,
            help='Path to the media folder to scan and process'
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
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be imported without processing'
        )
        parser.add_argument('--workers_clip', type=int, default=None, help="Workers for clip extraction")
        parser.add_argument('--workers_keyframes', type=int, default=None, help="Workers for keyframe extraction")

    def handle(self, *args, **options):
        media_path = options['media_path']
        dry_run = options['dry_run']
        
        # Step 1: Scan media folder
        self.stdout.write(self.style_info("=== Scanning Media Folder ==="))
        scan_options = {
            'media_path': media_path,
            'no_recursive': options['no_recursive'],
            'extensions': options['extensions'],
            'dry_run': dry_run
        }
        call_command("scan_media_folder", **scan_options)
        
        if dry_run:
            self.stdout.write(self.style_success("🧪 DRY RUN completed - no further processing"))
            return
            
        # Step 2: Full processing pipeline
        from multiprocessing import cpu_count
        worker_clip = options.get("workers_clip") or min(1, cpu_count())
        worker_keyframes = options.get("workers_keyframes") or min(1, cpu_count())

        self.stdout.write(self.style_info("=== Creating Web Proxies ==="))
        call_command("create_web_videos", max_height=480, quality=18)

        self.stdout.write(self.style_info("=== Extracting Clips ==="))
        call_command("extract_clips", workers=worker_clip)

        self.stdout.write(self.style_info("=== Extracting Keyframes ==="))
        keyframe_kwargs = {
            "search_range_factor": 0.95 if torch.cuda.is_available() else 0.5,
            "frames_to_compare": 50 if torch.cuda.is_available() else 5,
            "workers": worker_keyframes
        }
        call_command("extract_keyframes", **keyframe_kwargs)

        self.stdout.write(self.style_info("=== Extracting Objects ==="))
        call_command("extract_objects", batch_size=4)

        self.stdout.write(self.style_success("🎉 Complete import from media folder finished!"))
        self.stdout.write("🔍 Your videos are now ready for search and retrieval")