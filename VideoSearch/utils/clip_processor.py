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
        
        keyframes.extend(self._create_baseline_keyframes(clip))
        
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
            if Keyframe.objects.filter(clip=clip, frame=sentence_data['frame']).exists():
                if self.command:
                    self.command.stdout.write(f"Skipping sentence keyframe at frame {sentence_data['frame']} - already exists")
                continue
                
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
    
    def _create_baseline_keyframes(self, clip: Clip) -> List[Keyframe]:
        keyframes = []
        
        start_frame = clip.start_frame
        end_frame = clip.end_frame
        
        for frame in [start_frame, end_frame]:
            if Keyframe.objects.filter(clip=clip, frame=frame).exists():
                if self.command:
                    self.command.stdout.write(f"Skipping baseline keyframe at frame {frame} - already exists")
                continue
                
            try:
                keyframe = Keyframe.objects.create(
                    clip=clip,
                    frame=frame,
                    embedding_clip=b'',
                    transcript_text="",
                    transcript_confidence=0.0,
                    transcript_context="",
                    transcript_embedding=None
                )
                keyframes.append(keyframe)
                
                if self.command:
                    self.command.stdout.write(f"Created baseline keyframe at frame {frame}")
                    
            except Exception as e:
                if self.command:
                    self.command.stdout.write(f"Failed to create baseline keyframe at frame {frame}: {e}")
        
        return keyframes
    
    def _extract_all_features(self, keyframe: Keyframe):
        try:
            image = keyframe.clip.get_frame_image(keyframe.frame)
            if image is None:
                if self.command:
                    self.command.stdout.write(f"Failed to extract image for keyframe {keyframe.id} - deleting keyframe")
                keyframe.delete()
                return
            
            features = self.visual_extractor.extract_features(image)
            if not features or features.get('clip_emb') is None:
                if self.command:
                    self.command.stdout.write(f"Failed to extract CLIP embedding for keyframe {keyframe.id} - deleting keyframe")
                keyframe.delete()
                return
            
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
    
    def _fill_visual_gaps(self, clip: Clip, existing_keyframes: List[Keyframe]):
        if len(existing_keyframes) < 2:
            return
        
        # Get all existing keyframes including newly added ones
        all_keyframes = Keyframe.objects.filter(clip=clip).order_by('frame')
        keyframes_by_frame = list(all_keyframes)
        
        if len(keyframes_by_frame) < 2:
            return
        
        # Process gaps between adjacent keyframes recursively
        gaps_processed = 0
        for i in range(len(keyframes_by_frame) - 1):
            current = keyframes_by_frame[i]
            next_kf = keyframes_by_frame[i + 1]
            
            gaps_in_region = self._fill_visual_gaps_recursively(
                clip, 
                current.frame, 
                next_kf.frame
            )
            gaps_processed += gaps_in_region
        
        if self.command and gaps_processed > 0:
            self.command.stdout.write(f"Recursively filled {gaps_processed} visual gaps in clip {clip.id}")
    
    def _fill_visual_gaps_recursively(self, clip: Clip, start_frame: int, end_frame: int, depth: int = 0) -> int:
        """
        Recursively fill visual gaps between keyframes based purely on visual dissimilarity.
        Returns the number of gaps filled.
        """
        gaps_filled = 0
        
        gap_frames = end_frame - start_frame
        if gap_frames <= 6:
            return gaps_filled
        
        start_kf = Keyframe.objects.filter(clip=clip, frame=start_frame).first()
        end_kf = Keyframe.objects.filter(clip=clip, frame=end_frame).first()
        
        if start_kf and end_kf and self._visual_dissimilarity_high(start_kf, end_kf):
            midpoint_frame = (start_frame + end_frame) // 2
            
            if (midpoint_frame - start_frame < 3) or (end_frame - midpoint_frame < 3):
                return gaps_filled
                
            if Keyframe.objects.filter(clip=clip, frame=midpoint_frame).exists():
                return gaps_filled
            
            try:
                gap_keyframe = Keyframe.objects.create(
                    clip=clip,
                    frame=midpoint_frame,
                    embedding_clip=b'',
                    transcript_text="",
                    transcript_confidence=0.0,
                    transcript_context="",
                    transcript_embedding=None
                )
                
                self._extract_all_features(gap_keyframe)
                
                if Keyframe.objects.filter(clip=clip, frame=midpoint_frame).exists():
                    gaps_filled += 1
                    
                    if self.command:
                        self.command.stdout.write(f"Added recursive gap keyframe at frame {midpoint_frame} (depth {depth})")
                    
                    gaps_filled += self._fill_visual_gaps_recursively(
                        clip, start_frame, midpoint_frame, depth + 1
                    )
                    gaps_filled += self._fill_visual_gaps_recursively(
                        clip, midpoint_frame, end_frame, depth + 1
                    )
                
            except Exception as e:
                if self.command:
                    self.command.stdout.write(f"Failed to create recursive gap keyframe at frame {midpoint_frame}: {e}")
        
        return gaps_filled
    
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