from django.core.management.base import BaseCommand
from VideoSearch.management.base import StyledCommand as BaseCommand
from VideoSearch.models import Video
from pathlib import Path
import subprocess

class Command(BaseCommand):
    help = "Scan video directory (\'.data/videos/\') and import metadata into the database."

    def handle(self, *args, **kwargs):
        video_dir = Path('./data/videos')

        if not video_dir.exists():
            video_dir.mkdir(parents=True)
            self.stdout.write(self.style.WARNING(f"Directory '{video_dir}' did not exist, created it. Place videos into the folder and run the command again."))
            return

        video_files = [f for f in video_dir.iterdir() if is_valid_video(f)]

        if not video_files:
            self.stdout.write(self.style.WARNING(f"Directory '{video_dir}' is empty or contains no supported video files."))
            return

        for full_path in video_files:

            if Video.objects.filter(file_path=full_path).exists():
                self.stdout.write(self.style_info(f"Already imported: {full_path.name} - skipping."))
                continue

            meta = get_video_metadata(full_path)
            if not meta:
                self.stdout.write(self.style.WARNING(f"Could not read metadata for {full_path.name} - skipping."))
                continue

            video = Video.objects.create(
                    frame_count=meta['frame_count'],
                    fps_num=meta['fps_num'],
                    fps_den=meta['fps_den'],
                    resolution=f"{meta['width']}x{meta['height']}",
                    file_path=str(full_path),
                )
            self.stdout.write(self.style.SUCCESS(f"Imported {full_path.name} (ID {video.id})"))

def is_valid_video(file_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=codec_type', '-of', 'default=nw=1', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return 'codec_type=video' in result.stdout.decode()
    except Exception:
        return False

def get_video_metadata(file_path):
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(file_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode().strip().split('\n')

        width = int(output[0])
        height = int(output[1])
        fps_num, fps_den = map(int, output[2].split('/'))
        nb_frames = int(output[3]) if output[3].isdigit() else None

        return {
            'width': width,
            'height': height,
            'fps_num': fps_num,
            'fps_den': fps_den,
            'frame_count': nb_frames
        }
    except Exception as e:
        print(f"ffprobe failed: {e}")
        return None