import logging
from typing import List, Dict, Tuple, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class OCRExtractor:
    def __init__(self, languages=['en'], gpu=True, command=None):
        self.command = command
        self.languages = languages
        self.gpu = gpu
        self._reader = None
        
    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
                if self.command:
                    self.command.stdout.write("EasyOCR initialized successfully")
            except ImportError:
                if self.command:
                    self.command.stdout.write("EasyOCR not installed. Please run: pip install easyocr")
                raise ImportError("EasyOCR is required but not installed")
            except Exception as e:
                if self.command:
                    self.command.stdout.write(f"EasyOCR initialization failed: {e}")
                raise
        return self._reader
    
    def extract_text(self, image: Image.Image, confidence_threshold: float = 0.5) -> Dict:
        try:
            reader = self._get_reader()
            
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            results = reader.readtext(image_array)
            
            if not results:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'bboxes': []
                }
            
            filtered_results = [
                (bbox, text, conf) for bbox, text, conf in results 
                if conf >= confidence_threshold
            ]
            
            if not filtered_results:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'bboxes': []
                }
            
            texts = [text.strip() for _, text, _ in filtered_results]
            confidences = [conf for _, _, conf in filtered_results]
            bboxes = [bbox for bbox, _, _ in filtered_results]
            
            combined_text = ' '.join(texts).strip()
            avg_confidence = sum(confidences) / len(confidences)
            
            bbox_data = []
            for bbox, text, conf in filtered_results:
                bbox_data.append({
                    'bbox': bbox,
                    'text': text.strip(),
                    'confidence': conf
                })
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'bboxes': bbox_data
            }
            
        except Exception as e:
            if self.command:
                self.command.stdout.write(f"OCR extraction failed: {e}")
            logger.error(f"OCR extraction error: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'bboxes': []
            }
    
    def extract_with_embedding(self, image: Image.Image, confidence_threshold: float = 0.5) -> Dict:
        ocr_result = self.extract_text(image, confidence_threshold)
        
        if not ocr_result['text']:
            ocr_result['embedding'] = None
            return ocr_result
        
        try:
            from VideoSearch.utils.transcript_embeddings import get_transcript_embedder
            embedder = get_transcript_embedder()
            embedding = embedder.encode_text(ocr_result['text'])
            ocr_result['embedding'] = embedding
        except Exception as e:
            if self.command:
                self.command.stdout.write(f"OCR embedding failed: {e}")
            ocr_result['embedding'] = None
        
        return ocr_result