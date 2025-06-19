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
            self.stdout.write(self.style_warning(f"Directory '{video_dir}' did not exist, created it. Place videos into the folder and run the command again."))
            return

        video_files = [f for f in video_dir.rglob('*') if is_valid_video(f)]
        if not video_files:
            self.stdout.write(self.style_warning(f"Directory '{video_dir}' is empty or contains no supported video files."))
            return

        valid_paths = set(str(f.resolve()) for f in video_files)
        self.remove_stale_videos(valid_paths)

        for full_path in video_files:
            self.import_video_file(full_path)

    def remove_stale_videos(self, valid_paths):
        db_videos = list(Video.objects.all())
        deleted_filenames = []

        for video in db_videos:
            db_path = Path(video.file_path).resolve()   # Note: DB paths should be resolved but we are doing it again here just as a fallback if the paths are corrupted.
            if str(db_path) not in valid_paths:
                deleted_filenames.append(db_path.name)
                video.delete()
                continue

            meta = get_video_metadata(db_path)
            if not meta or not meta['frame_count']:
                deleted_filenames.append(db_path.name)
                video.delete()
                continue

            new_resolution = f"{meta['width']}x{meta['height']}"
            if (
                video.frame_count != meta['frame_count'] or
                video.fps_num != meta['fps_num'] or
                video.fps_den != meta['fps_den'] or
                video.resolution != new_resolution
            ):
                deleted_filenames.append(db_path.name)
                video.delete()
                continue

        if deleted_filenames:
            self.stdout.write(self.style_info(f"Removed {len(deleted_filenames)} video(s) no longer present:"))
            for name in deleted_filenames:
                self.stdout.write(self.style_info(f"  - {name}"))

    def import_video_file(self, full_path):
        resolved_path = str(full_path.resolve())
        if Video.objects.filter(file_path=resolved_path).exists():
            self.stdout.write(self.style_info(f"Already imported: {full_path.name} - skipping."))
            return

        meta = get_video_metadata(full_path)
        if not meta:
            self.stdout.write(self.style_warning(f"Could not read metadata for {full_path.name} - skipping."))
            return
        if not meta['frame_count']:
            self.stdout.write(self.style_warning(f"{full_path.name} has 0 frames - skipping."))
            return

        video = Video.objects.create(
            frame_count=meta['frame_count'],
            fps_num=meta['fps_num'],
            fps_den=meta['fps_den'],
            resolution=f"{meta['width']}x{meta['height']}",
            file_path=resolved_path,
        )
        self.stdout.write(self.style_success(f"Imported {full_path.name} (ID {video.id})"))

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