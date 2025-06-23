import os
from django.conf import settings
from django.db import models
from pathlib import Path

class Video(models.Model):
    frame_count = models.IntegerField()
    fps_num = models.IntegerField()
    fps_den = models.IntegerField()
    resolution = models.CharField(max_length=50)
    file_path = models.FilePathField(path="./data/videos/", max_length=500, unique=True)
    
    @property
    def media_url(self):
        relative_path = os.path.relpath(self.file_path, settings.MEDIA_ROOT)
        return settings.MEDIA_URL + relative_path.replace('\\', '/')
    
    @property
    def file_name(self):
        return Path(self.file_path).stem
    
    @property
    def fps(self) -> float:
        """Return the frames per second as a float."""
        return self.fps_num / self.fps_den if self.fps_den else 1.0

    def save(self, *args, **kwargs):
        self.file_path = str(Path(self.file_path).resolve())
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"Video {self.id}: {self.file_path} ({self.resolution}, {self.fps_num}/{self.fps_den}, {self.frame_count}f)"

class Clip(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    start_frame = models.IntegerField()
    end_frame = models.IntegerField()

    def __str__(self):
        return f"Clip {self.id}: Video {self.video} ({self.start_frame}f to {self.end_frame}f)"