from django.db import models
from pathlib import Path
import numpy as np
import zlib

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