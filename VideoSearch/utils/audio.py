import whisper
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class TranscriptSegment:
    """Represents a transcript segment with timing information."""
    start: float
    end: float
    text: str
    confidence: float
    words: List[Dict] = None

class AudioTranscriber:
    """Audio transcription using OpenAI Whisper."""
    
    def __init__(self, model_size: str = "base", command=None):
        """
        Initialize the transcriber.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            command: Django management command for output (optional)
        """
        self.model_size = model_size
        self.model = None
        self.command = command
    
    def _load_model(self):
        """Load the Whisper model on first use."""
        if self.model is None:
            if self.command:
                self.command.stdout.write(f"   Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
    
    def extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video using FFmpeg.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Path to extracted audio file (temporary)
        """
        video_path = Path(video_path)
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            audio_path = temp_audio.name
        
        # Extract audio using FFmpeg
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ac", "1",              # Mono
            "-ar", "16000",          # 16kHz sample rate
            "-y",                    # Overwrite output
            audio_path
        ]
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            return audio_path
        except subprocess.CalledProcessError as e:
            if self.command:
                self.command.stdout.write(
                    self.command.style_error(f"Audio extraction failed: {e.stderr}")
                )
            raise RuntimeError(f"Audio extraction failed: {e.stderr}")
    
    def transcribe_video(self, video_path: str) -> List[TranscriptSegment]:
        """
        Transcribe audio from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of transcript segments with timing
        """
        self._load_model()
        
        # Extract audio
        if self.command:
            self.command.stdout.write(f"   Extracting audio from {Path(video_path).name}")
        
        audio_path = self.extract_audio(video_path)
        
        try:
            # Transcribe with word-level timestamps
            if self.command:
                self.command.stdout.write("   Running Whisper transcription...")
                
            result = self.model.transcribe(
                audio_path,
                word_timestamps=True,
                verbose=False
            )
            
            segments = []
            for segment in result["segments"]:
                transcript_segment = TranscriptSegment(
                    start=segment["start"],
                    end=segment["end"],
                    text=segment["text"].strip(),
                    confidence=segment.get("avg_logprob", 0.0),
                    words=segment.get("words", [])
                )
                segments.append(transcript_segment)
            
            return segments
            
        finally:
            # Clean up temporary audio file
            Path(audio_path).unlink(missing_ok=True)
    
    def map_transcript_to_keyframe(
        self, 
        keyframe_time: float, 
        segments: List[TranscriptSegment],
        context_window: float = 5.0
    ) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        """
        Map transcript to a keyframe based on timestamp.
        
        Args:
            keyframe_time: Time of keyframe in seconds
            segments: List of transcript segments
            context_window: Seconds of context to include around keyframe
            
        Returns:
            Tuple of (primary_text, confidence, context_text)
        """
        if not segments:
            return None, None, None
            
        # Find segments that overlap with keyframe time
        primary_segments = []
        context_segments = []
        
        context_start = keyframe_time - context_window
        context_end = keyframe_time + context_window
        
        for segment in segments:
            # Check if segment overlaps with keyframe time (±1 second tolerance)
            if segment.start <= keyframe_time <= segment.end + 1.0:
                primary_segments.append(segment)
            # Check if segment is within context window
            elif context_start <= segment.start <= context_end or context_start <= segment.end <= context_end:
                context_segments.append(segment)
        
        # Build primary text and confidence
        primary_text = None
        primary_confidence = None
        
        if primary_segments:
            primary_text = " ".join(s.text for s in primary_segments)
            # Weight confidence by segment length
            total_duration = sum(s.end - s.start for s in primary_segments)
            if total_duration > 0:
                primary_confidence = sum(
                    s.confidence * (s.end - s.start) for s in primary_segments
                ) / total_duration
            else:
                primary_confidence = sum(s.confidence for s in primary_segments) / len(primary_segments)
        
        # Build context text with distance-based weighting
        context_parts = []
        
        # Add segments before keyframe
        before_segments = [s for s in context_segments if s.end < keyframe_time]
        before_segments.sort(key=lambda s: s.start)
        
        # Add segments after keyframe  
        after_segments = [s for s in context_segments if s.start > keyframe_time]
        after_segments.sort(key=lambda s: s.start)
        
        # Combine context
        all_context = before_segments + primary_segments + after_segments
        if all_context:
            context_text = " ".join(s.text for s in all_context)
        else:
            context_text = None
            
        return primary_text, primary_confidence, context_text

def keyframe_time_from_video_info(video, clip, keyframe_frame: int) -> float:
    """
    Calculate keyframe timestamp in seconds.
    
    Args:
        video: Video model instance
        clip: Clip model instance  
        keyframe_frame: Frame number relative to clip start
        
    Returns:
        Time in seconds
    """
    absolute_frame = clip.start_frame + keyframe_frame
    fps = video.fps()
    return absolute_frame / fps