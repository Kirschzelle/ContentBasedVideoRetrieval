from django.db import models

class Video(models.Model):
    duration = models.FloatField()
    fps_num = models.IntegerField()
    fps_den = models.IntegerField()
    resolution = models.CharField(max_length=50)
    file_path = models.FilePathField(path="./data/videos/", max_length=500, unique=True)
    
    def __str__(self):
        return f"Video {self.id}: {self.file_path} ({self.resolution}, {self.fps_num}/{self.fps_den}, {self.duration:.2f}s)"

class Clip(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    start_frame = models.IntegerField()
    end_frame = models.IntegerField()

    def __str__(self):
        return f"Clip {self.id}: Video {self.video} ({self.start_frame} to {self.end_frame})"