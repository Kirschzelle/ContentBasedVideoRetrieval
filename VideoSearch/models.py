import os
from django.conf import settings
from django.db import models
from pathlib import Path
import numpy as np
import zlib
import cv2
import subprocess
import tempfile
from PIL import Image

KEYFRAME_ROOT = Path("data/keyframes")

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

    def fps(self) -> float:
        """Return the frames per second as a float."""
        return self.fps_num / self.fps_den if self.fps_den else 1.0

    def video_duration(self) -> float:
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

    def get_selected_frame_images(self, frame_numbers: list[int], as_pil=True):
        images = {}
        cap = cv2.VideoCapture(str(self.file_path))

        for frame_index in sorted(set(frame_numbers)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = cap.read()
            if success and frame is not None:
                images[frame_index] = (
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if as_pil else frame
                )

        cap.release()

        return [images.get(f) for f in frame_numbers]

    def media_url(self):
        rel_path = os.path.relpath(self.file_path, settings.MEDIA_ROOT)
        return f"{settings.MEDIA_URL}{rel_path.replace(os.sep, '/')}"
    
    def __str__(self):
        return f"Video {self.id}: {self.file_path} ({self.resolution}, {self.fps_num}/{self.fps_den}, {self.frame_count}f)"

class Clip(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    start_frame = models.IntegerField()
    end_frame = models.IntegerField()

    def fps(self) -> float:
        return self.video.fps()

    def clip_duration(self) -> float:
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

    def get_selected_frame_images(self, relative_indices: list[int], as_pil: bool = True):
        """
        Loads only specific frame indices (relative to this clip).
        """
        absolute_indices = [
            self.start_frame + i for i in relative_indices
            if self.start_frame + i <= self.end_frame
        ]
        return self.video.get_selected_frame_images(absolute_indices, as_pil=as_pil)

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

    # Embeddings
    embedding_clip = models.BinaryField()
    embedding_dino = models.BinaryField(null=True, blank=True)

    # Color descriptors
    histogram_hsv = models.BinaryField(null=True, blank=True)
    dominant_colors = models.BinaryField(null=True, blank=True)
    colorfulness = models.FloatField(null=True, blank=True)  

    # Object
    object_vector = models.BinaryField(null=True, blank=True)

    class Meta:
        unique_together = ("clip", "frame")

    def __str__(self):
        return f"Keyframe {self.id}: Clip {self.clip.id} at frame {self.frame}"

    @staticmethod
    def compress_array(array: np.ndarray) -> bytes:
        return zlib.compress(array.astype(np.float32).tobytes())

    @staticmethod
    def decompress_array(blob: bytes, dtype=np.float32) -> np.ndarray:
        return np.frombuffer(zlib.decompress(blob), dtype=dtype).copy()
    
    def load_embedding_clip(self):
        return self.decompress_array(self.embedding_clip)

    def load_embedding_dino(self):
        return self.decompress_array(self.embedding_dino) if self.embedding_dino else None

    def load_histogram_hsv(self):
        return self.decompress_array(self.histogram_hsv) if self.histogram_hsv else None

    def load_dominant_colors(self):
        arr = self.decompress_array(self.dominant_colors) if self.dominant_colors else None
        return arr.reshape(-1, 3) if arr is not None else None

    def get_image_path(self) -> Path:
        """Returns the expected disk path for the keyframe image."""
        return KEYFRAME_ROOT / str(self.clip.id) / f"frame{self.frame}.jpg"

    def save_image(self):
        """Extracts and saves the keyframe image to disk."""
        img = self.clip.get_frame_image(self.frame)
        if img is None:
            return
        img_path = self.get_image_path()
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(img_path)

    def load_image(self) -> Image.Image | None:
        """Loads the saved keyframe image from disk."""
        path = self.get_image_path()
        return Image.open(path) if path.exists() else None

    def load_object_vector(self):
        return self.decompress_array(self.object_vector) if self.object_vector else None

    def get_features_from_keyframe(self) -> dict:
        return {
            "clip_emb": self.load_embedding_clip(),
            "dino_emb": self.load_embedding_dino(),
            "histogram": self.load_histogram_hsv(),
            "palette": self.load_dominant_colors(),
            "colorfulness": self.colorfulness,
            "object_vector": self.load_object_vector()
        }

    @classmethod
    def create(
        cls,
        clip,
        frame,
        embedding_clip: np.ndarray,
        embedding_dino: np.ndarray = None,
        histogram_hsv: np.ndarray = None,
        dominant_colors: np.ndarray = None,
        colorfulness: float = None,
        object_vector: dict = None
    ):
        keyframe = cls.objects.create(
            clip=clip,
            frame=frame,
            embedding_clip=cls.compress_array(embedding_clip),
            embedding_dino=cls.compress_array(embedding_dino) if embedding_dino is not None else None,
            histogram_hsv=cls.compress_array(histogram_hsv) if histogram_hsv is not None else None,
            dominant_colors=cls.compress_array(dominant_colors) if dominant_colors is not None else None,
            colorfulness=colorfulness,
            object_vector=cls.compress_array(object_vector) if object_vector is not None else None,
        )
        keyframe.save_image()
        return keyframe