from typing import List, Optional, Dict, Tuple
from VideoSearch.models import Clip, Keyframe
from VideoSearch.utils.audio import AudioTranscriber, TranscriptSegment
from VideoSearch.utils.visual_feature_extractor import VisualFeatureExtractor
from VideoSearch.utils.transcript_embeddings import get_transcript_embedder
from VideoSearch.utils.objects import ObjectDetector
from VideoSearch.utils.ocr_extractor import OCRExtractor
import numpy as np
from pathlib import Path

class ClipProcessor:
    def __init__(self, whisper_model_size: str = "base", command=None):
        self.command = command
        self.transcriber = AudioTranscriber(model_size=whisper_model_size, command=command)
        self.visual_extractor = VisualFeatureExtractor(command=command)
        self.transcript_embedder = get_transcript_embedder()
        self.object_extractor = ObjectDetector(command=command)
        self.ocr_extractor = OCRExtractor(command=command)
    
    def process_clip(self, clip: Clip) -> List[Keyframe]:
        if self.command:
            self.command.stdout.write(f"Processing clip {clip.id} (frames {clip.start_frame}-{clip.end_frame})")
        
        Keyframe.objects.filter(clip=clip).delete()
        
        sentences = self._extract_audio_sentences(clip)
        keyframes = self._create_sentence_keyframes(clip, sentences)
        
        for keyframe in keyframes:
            self._extract_all_features(keyframe)
        
        self._fill_visual_gaps(clip, keyframes)
        
        if self.command:
            final_count = Keyframe.objects.filter(clip=clip).count()
            self.command.stdout.write(f"Clip {clip.id}: {final_count} keyframes created")
        
        return Keyframe.objects.filter(clip=clip).all()
    
    def _extract_audio_sentences(self, clip: Clip) -> List[Dict]:
        try:
            video_path = Path(clip.video.processing_path)
            if not video_path.exists():
                if self.command:
                    self.command.stdout.write(f"Video file not found: {video_path}")
                return []
            
            segments = self.transcriber.transcribe_video(str(video_path))
            if not segments:
                return []
            
            clip_start_time = clip.start_frame / clip.video.fps()
            clip_end_time = clip.end_frame / clip.video.fps()
            
            sentences = []
            for i, segment in enumerate(segments):
                if segment.start >= clip_end_time or segment.end <= clip_start_time:
                    continue
                
                segment_time = max(segment.start, clip_start_time) - clip_start_time
                segment_frame = int(segment_time * clip.video.fps())
                
                prev_text = segments[i-1].text.strip() if i > 0 else ""
                next_text = segments[i+1].text.strip() if i < len(segments)-1 else ""
                
                sentences.append({
                    'frame': segment_frame,
                    'text': segment.text.strip(),
                    'confidence': segment.confidence,
                    'prev_text': prev_text,
                    'next_text': next_text,
                    'context': f"{prev_text} {segment.text.strip()} {next_text}".strip()
                })
            
            return sentences
            
        except Exception as e:
            if self.command:
                self.command.stdout.write(f"Audio extraction failed: {e}")
            return []
    
    def _create_sentence_keyframes(self, clip: Clip, sentences: List[Dict]) -> List[Keyframe]:
        keyframes = []
        for sentence_data in sentences:
            try:
                transcript_embedding = None
                if sentence_data['context']:
                    transcript_embedding = self.transcript_embedder.encode_text(sentence_data['context'])
                
                keyframe = Keyframe.objects.create(
                    clip=clip,
                    frame=sentence_data['frame'],
                    embedding_clip=b'',
                    transcript_text=sentence_data['text'],
                    transcript_confidence=sentence_data['confidence'],
                    transcript_context=sentence_data['context'],
                    transcript_embedding=Keyframe.compress_array(transcript_embedding) if transcript_embedding is not None else None
                )
                keyframes.append(keyframe)
                
            except Exception as e:
                if self.command:
                    self.command.stdout.write(f"Failed to create keyframe at frame {sentence_data['frame']}: {e}")
        
        return keyframes
    
    def _extract_all_features(self, keyframe: Keyframe):
        try:
            image = keyframe.clip.get_frame_image(keyframe.frame)
            if image is None:
                if self.command:
                    self.command.stdout.write(f"Failed to extract image for keyframe {keyframe.id}")
                return
            
            features = self.visual_extractor.extract_features(image)
            
            keyframe.embedding_clip = Keyframe.compress_array(features['clip_emb'])
            keyframe.embedding_dino = Keyframe.compress_array(features['dino_emb']) if features.get('dino_emb') is not None else None
            keyframe.histogram_hsv = Keyframe.compress_array(features['histogram']) if features.get('histogram') is not None else None
            keyframe.dominant_colors = Keyframe.compress_array(features['palette']) if features.get('palette') is not None else None
            keyframe.colorfulness = features.get('colorfulness')
            
            try:
                object_vector = self.object_extractor.extract_vector(image)
                keyframe.object_vector = Keyframe.compress_array(object_vector) if object_vector is not None else None
            except Exception as e:
                if self.command:
                    self.command.stdout.write(f"Object extraction failed for keyframe {keyframe.id}: {e}")
            
            try:
                ocr_result = self.ocr_extractor.extract_with_embedding(image, confidence_threshold=0.5)
                keyframe.ocr_text = ocr_result['text'] if ocr_result['text'] else None
                keyframe.ocr_confidence = ocr_result['confidence'] if ocr_result['text'] else None
                keyframe.ocr_bboxes = ocr_result['bboxes'] if ocr_result['bboxes'] else None
                keyframe.ocr_embedding = Keyframe.compress_array(ocr_result['embedding']) if ocr_result['embedding'] is not None else None
            except Exception as e:
                if self.command:
                    self.command.stdout.write(f"OCR extraction failed for keyframe {keyframe.id}: {e}")
            
            keyframe.save()
            keyframe.save_image()
            
        except Exception as e:
            if self.command:
                self.command.stdout.write(f"Feature extraction failed for keyframe {keyframe.id}: {e}")
    
    def _fill_visual_gaps(self, clip: Clip, existing_keyframes: List[Keyframe], gap_threshold_seconds: float = 2.0):
        if len(existing_keyframes) < 2:
            return
        
        keyframes_by_frame = sorted(existing_keyframes, key=lambda kf: kf.frame)
        fps = clip.video.fps()
        gap_threshold_frames = gap_threshold_seconds * fps
        
        gaps_to_fill = []
        for i in range(len(keyframes_by_frame) - 1):
            current = keyframes_by_frame[i]
            next_kf = keyframes_by_frame[i + 1]
            gap_frames = next_kf.frame - current.frame
            
            if gap_frames > gap_threshold_frames or self._visual_dissimilarity_high(current, next_kf):
                midpoint_frame = (current.frame + next_kf.frame) // 2
                gaps_to_fill.append(midpoint_frame)
        
        for frame in gaps_to_fill:
            try:
                gap_keyframe = Keyframe.objects.create(
                    clip=clip,
                    frame=frame,
                    embedding_clip=b'',
                    transcript_text="",
                    transcript_confidence=0.0,
                    transcript_context="",
                    transcript_embedding=None
                )
                self._extract_all_features(gap_keyframe)
                
                if self.command:
                    self.command.stdout.write(f"Added gap-filling keyframe at frame {frame}")
                    
            except Exception as e:
                if self.command:
                    self.command.stdout.write(f"Failed to create gap keyframe at frame {frame}: {e}")
    
    def _visual_dissimilarity_high(self, keyframe1: Keyframe, keyframe2: Keyframe, threshold: float = 0.9) -> bool:
        try:
            if not keyframe1.embedding_clip or not keyframe2.embedding_clip:
                return True
            
            emb1 = keyframe1.load_embedding_clip()
            emb2 = keyframe2.load_embedding_clip()
            
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return similarity < threshold
            
        except Exception:
            return True