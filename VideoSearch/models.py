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
import io

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
        Extract a frame using ffmpeg instead of OpenCV.
        This method is more robust for corrupted or complex videos.
        """
        fps = self.frame_rate
        time_sec = frame_index / fps

        command = [
            "ffmpeg",
            "-loglevel", "error",
            "-ss", str(time_sec),
            "-i", str(self.file_path),
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-"
        ]

        try:
            output = subprocess.check_output(command, stderr=subprocess.DEVNULL)
            img = Image.open(io.BytesIO(output))
            return img if as_pil else np.array(img)
        except subprocess.CalledProcessError:
            print(f"[ERROR] ffmpeg failed to extract frame {frame_index} at {time_sec}s from {self.file_path}")
            return None
        except Exception as e:
            print(f"[ERROR] Unexpected error extracting frame {frame_index} from {self.file_path}: {e}")
            return None

    def get_frame_range_images(self, start_frame: int, end_frame: int, as_pil=True) -> list:
        """
        Extracts a sequence of frames using ffmpeg (frame accurate).
        """
        if start_frame < 0 or end_frame >= self.frame_count or end_frame < start_frame:
            return []

        fps = self.frame_rate
        with tempfile.TemporaryDirectory() as tmpdir:
            out_pattern = Path(tmpdir) / "frame_%05d.png"
            cmd = [
                "ffmpeg",
                "-loglevel", "error",
                "-i", str(self.file_path),
                "-vf", f"select='between(n\\,{start_frame}\\,{end_frame})'",
                "-vsync", "0",
                str(out_pattern)
            ]

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"[ERROR] ffmpeg failed to extract frames {start_frame}-{end_frame} from {self.file_path}")
                return []

            images = sorted(Path(tmpdir).glob("frame_*.png"))
            if as_pil:
                return [Image.open(p).convert("RGB").copy() for p in images]
            else:
                return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in images]

    def get_selected_frame_images(self, frame_numbers: list[int], as_pil=True) -> list:
        """
        Extracts selected frames using ffmpeg by seeking to each one individually.
        """
        images = {}
        fps = self.frame_rate

        for frame_index in sorted(set(frame_numbers)):
            time_sec = frame_index / fps
            try:
                cmd = [
                    "ffmpeg",
                    "-loglevel", "error",
                    "-ss", str(time_sec),
                    "-i", str(self.file_path),
                    "-frames:v", "1",
                    "-f", "image2pipe",
                    "-vcodec", "png",
                    "-"
                ]
                output = subprocess.check_output(cmd)
                img = Image.open(io.BytesIO(output))
                images[frame_index] = img.convert("RGB") if as_pil else np.array(img)
            except Exception as e:
                print(f"[WARN] Failed to extract frame {frame_index} from {self.file_path}: {e}")
                images[frame_index] = None

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
        try:
            frame = self.video.get_frame_image(absolute_frame)
        except Exception as e:
            print(f"Error reading frame {absolute_frame} from video {self.video.file_path}: {e}")
        return frame

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

        try:
            frames = self.video.get_frame_range_images(absolute_start, absolute_end, as_pil=as_pil)
        except Exception as e:
            print(f"Error reading frame {absolute_start}-{absolute_end} from video {self.video.file_path}: {e}")
        return frames

    def get_selected_frame_images(self, relative_indices: list[int], as_pil: bool = True):
        """
        Loads only specific frame indices (relative to this clip).
        """
        absolute_indices = [
            self.start_frame + i for i in relative_indices
            if self.start_frame + i <= self.end_frame
        ]
        try:
            frames = self.video.get_selected_frame_images(absolute_indices, as_pil=as_pil)
        except Exception as e:
            print(f"Error reading frames {absolute_indices} from video {self.video.file_path}: {e}")
            frames = []
        return frames

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