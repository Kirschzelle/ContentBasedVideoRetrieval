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

class MediaFolderSetting(models.Model):
    """Store media folder configuration"""
    name = models.CharField(max_length=100, unique=True, help_text="Descriptive name for this media folder")
    path = models.CharField(max_length=500, help_text="Full path to the media folder")
    recursive = models.BooleanField(default=True, help_text="Scan subdirectories recursively")
    extensions = models.CharField(
        max_length=200, 
        default="mp4,mov,mkv,avi,wmv,flv,webm,m4v",
        help_text="Comma-separated list of video file extensions"
    )
    is_active = models.BooleanField(default=True, help_text="Whether to include this folder in scans")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return f"{self.name}: {self.path}"
    
    @property
    def extension_list(self):
        """Return extensions as a list"""
        return [ext.strip().lower() for ext in self.extensions.split(',')]
    
    def path_exists(self):
        """Check if the folder path exists"""
        return Path(self.path).exists()

class Video(models.Model):
    frame_count = models.IntegerField()
    fps_num = models.IntegerField()
    fps_den = models.IntegerField()
    resolution = models.CharField(max_length=50)
    file_path = models.FilePathField(max_length=500, unique=True)
    web_path = models.FilePathField(path="./data/videos_web/", max_length=500, null=True, blank=True)
    
    @property
    def media_url(self):
        # Use web_path if available, otherwise fallback to original
        video_path = self.web_path if self.web_path and Path(self.web_path).exists() else self.file_path
        relative_path = os.path.relpath(video_path, settings.MEDIA_ROOT)
        return settings.MEDIA_URL + relative_path.replace('\\', '/')
    
    @property
    def file_name(self):
        return Path(self.file_path).stem
    
    @property
    def processing_path(self):
        """Return the best path for video processing (web proxy if available, otherwise original)"""
        if self.web_path and Path(self.web_path).exists():
            return self.web_path
        return self.file_path
    
    def save(self, *args, **kwargs):
        self.file_path = str(Path(self.file_path).resolve())
        if self.web_path:
            self.web_path = str(Path(self.web_path).resolve())
        super().save(*args, **kwargs)
    
    def delete(self, *args, **kwargs):
        # Delete associated web file when video is deleted
        if self.web_path and Path(self.web_path).exists():
            Path(self.web_path).unlink()
        super().delete(*args, **kwargs)

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
        Uses web proxy for faster processing when available, falls back to original if needed.
        """
        fps = self.fps()
        time_sec = frame_index / fps

        # Try with processing path first (prefers web video), then fallback to original
        for video_path in [self.processing_path, self.file_path]:
            command = [
                "ffmpeg",
                "-loglevel", "error",
                "-ss", str(time_sec),
                "-i", str(video_path),
                "-frames:v", "1",
                "-f", "image2pipe",
                "-vcodec", "png",
                "-"
            ]

            try:
                output = subprocess.check_output(command, stderr=subprocess.DEVNULL)
                img = Image.open(io.BytesIO(output))
                return img if as_pil else np.array(img)
            except Exception as e:
                if video_path == self.processing_path and video_path != self.file_path:
                    # Only warn on first attempt if we have a fallback
                    print(f"[WARN] Failed to extract frame {frame_index} from web video, trying original: {e}")
                else:
                    # Final failure
                    print(f"[ERROR] Failed to extract frame {frame_index} from {video_path}: {e}")
                    
        return None

    def get_frame_range_images(self, start_frame: int, end_frame: int, as_pil=True) -> list:
        """
        Extracts a sequence of frames using ffmpeg (frame accurate).
        Uses web proxy for faster processing when available, falls back to original if needed.
        """
        if start_frame < 0 or end_frame >= self.frame_count or end_frame < start_frame:
            return []

        fps = self.fps()
        
        # Try with processing path first (prefers web video), then fallback to original
        for video_path in [self.processing_path, self.file_path]:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_pattern = Path(tmpdir) / "frame_%05d.png"
                cmd = [
                    "ffmpeg",
                    "-loglevel", "error",
                    "-i", str(video_path),
                    "-vf", f"select='between(n\\,{start_frame}\\,{end_frame})'",
                    "-vsync", "0",
                    str(out_pattern)
                ]

                try:
                    subprocess.run(cmd, check=True)
                    images = sorted(Path(tmpdir).glob("frame_*.png"))
                    if as_pil:
                        return [Image.open(p).convert("RGB").copy() for p in images]
                    else:
                        return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in images]
                except subprocess.CalledProcessError as e:
                    if video_path == self.processing_path and video_path != self.file_path:
                        # Only warn on first attempt if we have a fallback
                        print(f"[WARN] Failed to extract frames {start_frame}-{end_frame} from web video, trying original: {e}")
                    else:
                        # Final failure
                        print(f"[ERROR] Failed to extract frames {start_frame}-{end_frame} from {video_path}: {e}")
                        
        return []

    def get_selected_frame_images(self, frame_numbers: list[int], as_pil=True) -> list:
        """
        Extracts selected frames using ffmpeg by seeking to each one individually.
        Uses web proxy for faster processing when available, falls back to original if needed.
        """
        images = {}
        fps = self.fps()

        for frame_index in sorted(set(frame_numbers)):
            time_sec = frame_index / fps
            
            # Try with processing path first (prefers web video)
            success = False
            for video_path in [self.processing_path, self.file_path]:
                if success:
                    break
                    
                try:
                    cmd = [
                        "ffmpeg",
                        "-loglevel", "error",
                        "-ss", str(time_sec),
                        "-i", str(video_path),
                        "-frames:v", "1",
                        "-f", "image2pipe",
                        "-vcodec", "png",
                        "-"
                    ]
                    output = subprocess.check_output(cmd)
                    img = Image.open(io.BytesIO(output))
                    images[frame_index] = img.convert("RGB") if as_pil else np.array(img)
                    success = True
                except Exception as e:
                    if video_path == self.processing_path and video_path != self.file_path:
                        # Only warn on first attempt if we have a fallback
                        print(f"[WARN] Failed to extract frame {frame_index} from web video, trying original: {e}")
                    else:
                        # Final failure (either no fallback available or fallback also failed)
                        print(f"[ERROR] Failed to extract frame {frame_index} from {video_path}: {e}")
                        images[frame_index] = None

        return [images.get(f) for f in frame_numbers]
    
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

    # Audio transcript
    transcript_text = models.TextField(null=True, blank=True)
    transcript_confidence = models.FloatField(null=True, blank=True) 
    transcript_context = models.TextField(null=True, blank=True)
    transcript_embedding = models.BinaryField(null=True, blank=True)

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

    def load_transcript_embedding(self):
        return self.decompress_array(self.transcript_embedding) if self.transcript_embedding else None

    def get_transcript_data(self) -> dict:
        """Returns transcript data for this keyframe."""
        return {
            "text": self.transcript_text,
            "confidence": self.transcript_confidence,
            "context": self.transcript_context,
            "embedding": self.load_transcript_embedding()
        }

    def get_features_from_keyframe(self) -> dict:
        return {
            "clip_emb": self.load_embedding_clip(),
            "dino_emb": self.load_embedding_dino(),
            "histogram": self.load_histogram_hsv(),
            "palette": self.load_dominant_colors(),
            "colorfulness": self.colorfulness,
            "object_vector": self.load_object_vector(),
            "transcript_embedding": self.load_transcript_embedding(),
            "transcript": self.get_transcript_data()
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
        object_vector: dict = None,
        transcript_text: str = None,
        transcript_confidence: float = None,
        transcript_context: str = None,
        transcript_embedding: np.ndarray = None
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
            transcript_text=transcript_text,
            transcript_confidence=transcript_confidence,
            transcript_context=transcript_context,
            transcript_embedding=cls.compress_array(transcript_embedding) if transcript_embedding is not None else None
        )
        keyframe.save_image()
        return keyframe

# Analytics models for smart weighting system
class SearchInteraction(models.Model):
    """Track user search interactions for smart weighting system."""
    
    # Search context
    query = models.TextField()
    query_type = models.CharField(
        max_length=20, 
        choices=[('balanced', 'Balanced'), ('visual', 'Visual'), ('text', 'Text')],
        default='balanced'
    )
    search_timestamp = models.DateTimeField(auto_now_add=True)
    
    # User context (optional for anonymous usage)
    session_id = models.CharField(max_length=64, null=True, blank=True)
    
    # Search parameters 
    had_filters = models.BooleanField(default=False)
    filter_types = models.TextField(null=True, blank=True)  # JSON list of filter types used
    
    # Results and interaction
    total_results_returned = models.IntegerField()
    results_keyframe_ids = models.TextField()  # JSON list of returned keyframe IDs in order
    
    class Meta:
        indexes = [
            models.Index(fields=['search_timestamp']),
            models.Index(fields=['query_type']),
            models.Index(fields=['had_filters']),
        ]

class ResultEngagement(models.Model):
    """Track detailed user engagement with search results."""
    
    search_interaction = models.ForeignKey(SearchInteraction, on_delete=models.CASCADE)
    keyframe = models.ForeignKey(Keyframe, on_delete=models.CASCADE)
    
    # Basic interaction
    result_position = models.IntegerField()  # 0-based position in search results
    was_clicked = models.BooleanField(default=False)
    click_timestamp = models.DateTimeField(null=True, blank=True)
    time_to_click = models.FloatField(null=True, blank=True)  # Seconds from search to click
    
    # Engagement quality indicators
    view_duration = models.FloatField(default=0.0)  # Total time spent viewing
    hover_duration = models.FloatField(default=0.0)  # Time hovering over result
    scroll_pauses = models.IntegerField(default=0)  # Times user paused scrolling on this result
    
    # Usage signals
    video_played = models.BooleanField(default=False)
    video_watch_duration = models.FloatField(default=0.0)  # Seconds of video watched
    video_completion_rate = models.FloatField(default=0.0)  # 0.0-1.0, how much of clip was watched
    
    # Action signals (realistic for video search)
    was_downloaded = models.BooleanField(default=False)  # Legacy field - will be replaced
    was_shared = models.BooleanField(default=False)  # Legacy field - will be replaced
    was_bookmarked = models.BooleanField(default=False)  # Legacy field - will be replaced
    clip_copied = models.BooleanField(default=False)  # Copied timestamp/clip info
    video_opened_fullscreen = models.BooleanField(default=False)  # Opened video in fullscreen
    was_used_as_filter = models.BooleanField(default=False)  # Used to create new filter
    video_scrubbed = models.BooleanField(default=False)  # User scrubbed through video timeline
    related_searches_performed = models.IntegerField(default=0)  # Searches based on this result
    
    # Navigation signals
    opened_in_new_tab = models.BooleanField(default=False)
    returned_to_result = models.BooleanField(default=False)  # Came back to this result later
    
    # Final outcome
    next_action = models.CharField(
        max_length=25,
        choices=[
            ('new_search', 'New Search'),
            ('refine_search', 'Refine Search'), 
            ('apply_filter', 'Apply Filter'),
            ('copy_clip', 'Copy Clip Info'),
            ('open_fullscreen', 'Open Fullscreen'),
            ('scrub_timeline', 'Scrub Timeline'),
            ('related_search', 'Search for Similar'),
            ('continue_browsing', 'Continue Browsing'),
            ('exit', 'Exit')
        ],
        null=True, blank=True
    )
    
    # Computed engagement score (updated when engagement data changes)
    engagement_score = models.FloatField(default=0.0)  # 0.0-1.0 quality score
    
    def calculate_engagement_score(self) -> float:
        """
        Calculate engagement quality score based on multiple signals.
        Returns score between 0.0 (poor engagement) and 1.0 (excellent engagement).
        """
        score = 0.0
        
        # Base click signal (weak positive signal)
        if self.was_clicked:
            score += 0.1
            
        # View time signals (strongest predictor)
        if self.view_duration > 0:
            if self.view_duration >= 30:  # 30+ seconds = very engaged
                score += 0.4
            elif self.view_duration >= 10:  # 10-30 seconds = moderately engaged  
                score += 0.25
            elif self.view_duration >= 3:   # 3-10 seconds = briefly engaged
                score += 0.1
                
        # Video engagement (very strong signal)
        if self.video_played:
            score += 0.2
            if self.video_completion_rate >= 0.8:  # Watched 80%+ of video
                score += 0.3
            elif self.video_completion_rate >= 0.5:  # Watched 50%+ of video
                score += 0.2
            elif self.video_completion_rate >= 0.2:  # Watched 20%+ of video
                score += 0.1
                
        # Action signals (realistic for video search - strong positive indicators)
        if self.clip_copied:
            score += 0.25  # User found it useful enough to copy info
        if self.video_opened_fullscreen:
            score += 0.3  # Strong interest signal
        if self.was_used_as_filter:
            score += 0.35  # Very strong - user found it useful for further searching
        if self.video_scrubbed:
            score += 0.15  # Active exploration of content
        if self.related_searches_performed > 0:
            score += min(0.2, self.related_searches_performed * 0.05)  # Inspired follow-up searches
            
        # Attention signals
        if self.hover_duration >= 2.0:  # Hovered for 2+ seconds
            score += 0.1
        if self.scroll_pauses >= 2:  # Paused scrolling multiple times
            score += 0.05
            
        # Navigation signals
        if self.opened_in_new_tab:
            score += 0.15
        if self.returned_to_result:
            score += 0.1
            
        # Position bias correction (results lower in list need higher engagement)
        position_penalty = min(0.1, self.result_position * 0.02)
        score = max(0.0, score - position_penalty)
        
        # Quick exit penalty (clicked but immediately left)
        if self.was_clicked and self.view_duration < 1.0 and not any([
            self.video_played, self.clip_copied, self.video_opened_fullscreen, self.was_used_as_filter
        ]):
            score *= 0.3  # Heavily penalize quick exits
            
        return min(1.0, score)  # Cap at 1.0
    
    def save(self, *args, **kwargs):
        """Update engagement score when saving."""
        self.engagement_score = self.calculate_engagement_score()
        super().save(*args, **kwargs)
    
    class Meta:
        unique_together = ('search_interaction', 'keyframe')
        indexes = [
            models.Index(fields=['result_position']),
            models.Index(fields=['engagement_score']),
            models.Index(fields=['was_clicked']),
            models.Index(fields=['video_played']),
        ]

class WeightingModel(models.Model):
    """Store learned weighting parameters for different query types."""
    
    query_type = models.CharField(max_length=20, unique=True)
    
    # Core feature weights (learned from user interactions)
    clip_weight = models.FloatField(default=1.0)
    transcript_weight = models.FloatField(default=0.8) 
    dino_weight = models.FloatField(default=0.6)
    object_weight = models.FloatField(default=0.4)
    histogram_weight = models.FloatField(default=0.3)
    
    # Metadata
    training_samples = models.IntegerField(default=0)
    last_updated = models.DateTimeField(auto_now=True)
    model_version = models.CharField(max_length=20, default='1.0')
    
    # Performance metrics
    click_through_rate = models.FloatField(null=True, blank=True)
    average_engagement_score = models.FloatField(null=True, blank=True)  # 0.0-1.0, higher is better
    high_engagement_rate = models.FloatField(null=True, blank=True)  # % of results with score > 0.5
    video_completion_rate = models.FloatField(null=True, blank=True)  # Average video completion rate
    average_result_position = models.FloatField(null=True, blank=True)  # Lower is better
    
    def get_weights_dict(self):
        """Return weights as dictionary compatible with CombinedVectorBuilder."""
        return {
            'clip_emb': self.clip_weight,
            'transcript_embedding': self.transcript_weight,
            'dino_emb': self.dino_weight, 
            'object_vector': self.object_weight,
            'histogram': self.histogram_weight,
        }
    
    def update_weights(self, new_weights: dict):
        """Update weights from training results."""
        self.clip_weight = new_weights.get('clip_emb', self.clip_weight)
        self.transcript_weight = new_weights.get('transcript_embedding', self.transcript_weight)
        self.dino_weight = new_weights.get('dino_emb', self.dino_weight)
        self.object_weight = new_weights.get('object_vector', self.object_weight)
        self.histogram_weight = new_weights.get('histogram', self.histogram_weight)
        self.save()
    
    class Meta:
        ordering = ['-last_updated']
    
    def __str__(self):
        return f"WeightingModel({self.query_type}) - {self.training_samples} samples"

class UploadedFrame(models.Model):
    """Store uploaded external frames for search filtering."""
    
    # File info
    filename = models.CharField(max_length=255)
    file_size = models.IntegerField()
    content_type = models.CharField(max_length=100)
    upload_timestamp = models.DateTimeField(auto_now_add=True)
    
    # Image metadata
    width = models.IntegerField()
    height = models.IntegerField()
    
    # Features (same as Keyframe)
    embedding_clip = models.BinaryField()
    embedding_dino = models.BinaryField(null=True, blank=True)
    histogram_hsv = models.BinaryField(null=True, blank=True)
    dominant_colors = models.BinaryField(null=True, blank=True)
    colorfulness = models.FloatField(null=True, blank=True)
    object_vector = models.BinaryField(null=True, blank=True)
    
    # Session tracking (optional)
    session_key = models.CharField(max_length=64, null=True, blank=True)
    
    class Meta:
        ordering = ['-upload_timestamp']  # Latest first
        indexes = [
            models.Index(fields=['upload_timestamp']),
            models.Index(fields=['session_key']),
        ]
    
    def __str__(self):
        return f"UploadedFrame: {self.filename} ({self.upload_timestamp})"
    
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

    def load_object_vector(self):
        return self.decompress_array(self.object_vector) if self.object_vector else None
    
    def get_features_dict(self) -> dict:
        """Return features in same format as Keyframe."""
        return {
            "clip_emb": self.load_embedding_clip(),
            "dino_emb": self.load_embedding_dino(),
            "histogram": self.load_histogram_hsv(),
            "palette": self.load_dominant_colors(),
            "colorfulness": self.colorfulness,
            "object_vector": self.load_object_vector(),
        }
    
    # Fake properties to make it compatible with search results display
    @property
    def id(self):
        return f"uploaded_{super().id}"  # Prefix to distinguish from real keyframes
    
    def get_image_path(self) -> Path:
        """Return a fake path for display purposes."""
        return Path(f"uploaded_frames/{self.filename}")
        
    @property
    def clip(self):
        """Fake clip property for template compatibility."""
        class FakeClip:
            @property
            def video(self):
                class FakeVideo:
                    @property 
                    def file_name(self):
                        return "Uploaded Frame"
                return FakeVideo()
        return FakeClip()