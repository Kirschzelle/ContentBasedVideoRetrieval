from django.db import models
from pathlib import Path
import numpy as np
import zlib
import cv2
import subprocess
import tempfile
from PIL import Image

class Video(models.Model):
    frame_count = models.IntegerField()
    fps_num = models.IntegerField()
    fps_den = models.IntegerField()
    resolution = models.CharField(max_length=50)
    file_path = models.FilePathField(path="./data/videos/", max_length=500, unique=True)

    def save(self, *args, **kwargs):
        self.file_path = str(Path(self.file_path).resolve())
        super().save(*args, **kwargs)

    def fps(self) -> float:
        """Return the frames per second as a float."""
        return self.fps_num / self.fps_den if self.fps_den else 1.0

    def duration(self) -> float:
        """Returns the duration of the video in seconds as a float."""
        return self.duration(self.frame_count)

    def duration(self, frames) -> float:
        """Returns the duration in seconds as a float."""
        return frames / self.fps()

    def get_frame_image(self, frame_index: int, as_pil: bool = True):
        """
        Extract a frame by absolute frame index.

        :param frame_index: Absolute frame number in the video.
        :param as_pil: If True, returns the frame as a `PIL.Image.Image`; otherwise as a raw BGR `np.ndarray`.
        :return: `PIL.Image.Image` or `np.ndarray` if successful, otherwise `None`.
        """
        if frame_index < 0 or frame_index >= self.frame_count:
            return None

        cap = cv2.VideoCapture(str(self.file_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        cap.release()

        if not success or frame is None:
            return None

        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if as_pil else frame

    def get_frame_range_images(self, start_frame: int, end_frame: int, as_pil=True) -> list:
        """
        Efficiently extracts a sequence of frames using ffmpeg.

        :param start_frame: First frame index (inclusive).
        :param end_frame: Last frame index (inclusive).
        :param as_pil: Return PIL images or raw file paths.
        :return: List of images as `PIL.Image.Image` if `as_pil=True`, or `np.ndarray` in BGR format otherwise.
        """
        if start_frame < 0 or end_frame >= self.frame_count or end_frame < start_frame:
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            out_pattern = Path(tmpdir) / "frame_%04d.jpg"
            cmd = [
                "ffmpeg",
                "-i", str(self.file_path),
                "-vf", f"select='between(n\\,{start_frame}\\,{end_frame})'",
                "-vsync", "0",
                "-q:v", "2",
                str(out_pattern),
                "-loglevel", "quiet"
            ]
            subprocess.run(cmd, check=True)

            images = sorted(Path(tmpdir).glob("frame_*.jpg"))
            if as_pil:
                loaded_images = []
                for img_path in images:
                    with Image.open(img_path) as im:
                        loaded_images.append(im.convert("RGB").copy())
                return loaded_images
            else:
                return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in images]
    
    def __str__(self):
        return f"Video {self.id}: {self.file_path} ({self.resolution}, {self.fps_num}/{self.fps_den}, {self.frame_count}f)"

class Clip(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    start_frame = models.IntegerField()
    end_frame = models.IntegerField()

    def fps(self) -> float:
        return self.video.fps()

    def duration(self) -> float:
        return self.duration(self.total_frames())

    def duration(self, frames) -> float:
        return self.video.duration(frames)

    def total_frames(self) -> int:
        return self.end_frame - self.start_frame

    def get_frame_image(self, offset: int = 0, as_pil: bool = True):
        """
        Get a frame within the clip range, relative to start_frame.
        """
        absolute_frame = self.start_frame + offset
        if absolute_frame > self.end_frame:
            return None
        return self.video.get_frame_image(absolute_frame, as_pil=as_pil)

    def get_frame_range_images(self, start: int = 0, end: int = None, as_pil: bool = True):
        """
        Retrieve a range of frames relative to the start of this clip.

        :param start: Start frame index (relative to clip start), inclusive.
        :param end: End frame index (relative to clip start), exclusive. Defaults to clip length.
        :param as_pil: If True, returns list of PIL.Image.Image; otherwise list of raw BGR np.ndarrays.
        :return: List of frame images.
        """
        if end is None:
            end = self.total_frames()

        absolute_start = self.start_frame + start
        absolute_end = self.start_frame + end

        return self.video.get_frame_range_images(absolute_start, absolute_end, as_pil=as_pil)

    def __str__(self):
        return f"Clip {self.id}: Video {self.video} ({self.start_frame}f to {self.end_frame}f)"

class ClipPredictionCache(models.Model):
    clip = models.OneToOneField(Clip, on_delete=models.CASCADE)
    probabilities = models.BinaryField() 

    @staticmethod
    def compress_array(array: np.ndarray) -> bytes:
        return zlib.compress(array.astype(np.float32).tobytes())

    @staticmethod
    def decompress_array(blob: bytes, dtype=np.float32) -> np.ndarray:
        return np.frombuffer(zlib.decompress(blob), dtype=dtype)

    @classmethod
    def store(cls, clip: Clip, predictions: np.ndarray):
        compressed = cls.compress_array(predictions)
        return cls.objects.update_or_create(clip=clip, defaults={"probabilities": compressed})[0]

    def load_predictions(self) -> np.ndarray:
        return self.decompress_array(self.probabilities)

class Keyframe(models.Model):
    clip = models.ForeignKey(Clip, on_delete=models.CASCADE)
    frame = models.IntegerField()
    embedding_clip = models.BinaryField()
    embedding_dino = models.BinaryField(null=True, blank=True)

    class Meta:
        unique_together = ("clip", "frame")

    def __str__(self):
        return f"Keyframe {self.id}: Clip {self.clip.id} at frame {self.frame}"

    @staticmethod
    def compress_array(array: np.ndarray) -> bytes:
        return zlib.compress(array.astype(np.float32).tobytes())

    @staticmethod
    def decompress_array(blob: bytes, dtype=np.float32) -> np.ndarray:
        return np.frombuffer(zlib.decompress(blob), dtype=dtype)

    @classmethod
    def create(cls, clip, frame, embedding_clip: np.ndarray, embedding_dino: np.ndarray = None):
        compressed_clip = cls.compress_array(embedding_clip)
        compressed_dino = cls.compress_array(embedding_dino) if embedding_dino is not None else None

        return cls.objects.create(
            clip=clip,
            frame=frame,
            embedding_clip=compressed_clip,
            embedding_dino=compressed_dino
        )

    def load_embedding_clip(self):
        return self.decompress_array(self.embedding_clip)

    def load_embedding_dino(self):
        return self.decompress_array(self.embedding_dino) if self.embedding_dino else None