from asyncio.windows_events import INFINITE
import time
import torch
import numpy as np
import logging
from transformers import CLIPTokenizer, CLIPModel
from VideoSearch.models import Keyframe
from VideoSearch.utils.hardware import EmbeddingModelSelector
from VideoSearch.utils.visual_feature_extractor import compute_distance, nonlinear_pooling
import utils.filters as ufil
from utils.annoy_index import build_annoy_index

logger = logging.getLogger(__name__)

class Searcher:
    def __init__(self):
        hardware = EmbeddingModelSelector()
        clip_model_name, _, _ = hardware.select()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            logger.info(f"Loading {clip_model_name} from local cache")
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name, local_files_only=True)
            try:
                self.model = CLIPModel.from_pretrained(clip_model_name, local_files_only=True).to(self.device)
            except NotImplementedError:
                model = CLIPModel.from_pretrained(clip_model_name, local_files_only=True)
                self.model = model.to_empty(device=self.device)
                self.model.load_state_dict(model.state_dict())
        except (OSError, ValueError):
            logger.info(f"Local cache not found, downloading {clip_model_name}")
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
            try:
                self.model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
            except NotImplementedError:
                model = CLIPModel.from_pretrained(clip_model_name)
                self.model = model.to_empty(device=self.device)
                self.model.load_state_dict(model.state_dict())

        self.last_query = None
        self.last_embedding = None
        self.last_query_objects = None

        self.kf_lookup = {
            kf.id: kf for kf in Keyframe.objects.select_related("clip", "clip__video")
            .only("id", "clip__video__id", "clip__id", "frame",
                  "embedding_clip", "embedding_dino", "histogram_hsv",
                  "dominant_colors", "colorfulness", "object_vector",
                  "transcript_text", "transcript_confidence", "transcript_context", "transcript_embedding",
                  "ocr_text", "ocr_confidence", "ocr_bboxes", "ocr_embedding")
        }

        self.clip_index, self.id_map = build_annoy_index(
            feature_name="clip_emb",
            kf_lookup=self.kf_lookup
        )

        self.dino_index, self.dino_id_map = build_annoy_index(
            feature_name="dino_emb",
            kf_lookup=self.kf_lookup
        )

        self.color_index, self.color_id_map = build_annoy_index(
            feature_name="histogram",
            kf_lookup=self.kf_lookup
        )

        self.object_index, self.object_id_map = build_annoy_index(
            feature_name="object_vector",
            kf_lookup=self.kf_lookup
        )

        self.ocr_index, self.ocr_id_map = build_annoy_index(
            feature_name="ocr_embedding",
            kf_lookup=self.kf_lookup
        )

    def search_streaming(self, query: str, search_mode: str = "balanced", 
                        session_state=None, filters=None, batch_size=10):
        if filters is None:
            filters = {}
        
        if session_state is None:
            session_state = {
                'buffers': {},
                'positions': {},
                'returned_ids': set()
            }
        
        active_indices = self._get_active_indices(search_mode, filters)
        
        if not session_state['buffers']:
            session_state = self._initialize_search_buffers(query, active_indices, session_state)
        
        results = []
        for _ in range(batch_size):
            next_result = self._get_next_best_result(session_state, filters, query)
            if next_result is None:
                break
            results.append(next_result)
            session_state['returned_ids'].add(next_result.id)
        
        done = all(len(buffer) == 0 for buffer in session_state['buffers'].values())
        
        return {
            'results': results,
            'session_state': session_state,
            'done': done
        }
    
    def _get_active_indices(self, search_mode: str, filters: dict) -> dict:
        indices = {}
        
        if search_mode in ["visual", "balanced"]:
            indices['clip'] = (self.clip_index, self.id_map)
            indices['colors'] = (self.color_index, self.color_id_map)
            indices['objects'] = (self.object_index, self.object_id_map)
            indices['ocr_text'] = None
            indices['ocr_embedding'] = (self.ocr_index, self.ocr_id_map)
            
        if search_mode in ["audio", "balanced"]:
            indices['transcript_text'] = None
            
        if filters:
            for kf_id, categories in filters.items():
                if str(kf_id).startswith('davinci_'):
                    continue
                for category in categories:
                    if category == "embeddings":
                        indices['dino'] = (self.dino_index, self.dino_id_map)
                        indices['clip'] = (self.clip_index, self.id_map)
                    elif category == "colors":
                        indices['colors'] = (self.color_index, self.color_id_map)
                    elif category == "objects":
                        indices['objects'] = (self.object_index, self.object_id_map)
        
        return indices
    
    def _initialize_search_buffers(self, query: str, active_indices: dict, session_state: dict) -> dict:
        buffer_size = 500
        query_embedding = self.encode_text(query)
        
        for index_name, index_data in active_indices.items():
            if index_name == "transcript_text":
                text_results = self._search_transcripts(query, threshold=-2.0)
                text_keyframes = [self.kf_lookup[kf_id] for kf_id in text_results if kf_id in self.kf_lookup]
                session_state['buffers'][index_name] = text_keyframes[:buffer_size]
                session_state['positions'][index_name] = len(text_keyframes)
            elif index_name == "ocr_text":
                ocr_results = self._search_ocr_text(query, threshold=0.5)
                ocr_keyframes = [self.kf_lookup[kf_id] for kf_id in ocr_results if kf_id in self.kf_lookup]
                session_state['buffers'][index_name] = ocr_keyframes[:buffer_size]
                session_state['positions'][index_name] = len(ocr_keyframes)
            elif index_name in ["dino", "colors", "objects", "ocr_embedding"]:
                session_state['buffers'][index_name] = []
                session_state['positions'][index_name] = 0
            else:
                index, id_map = index_data
                annoy_ids = index.get_nns_by_vector(query_embedding, buffer_size)
                keyframes = [self.kf_lookup[id_map[i]] for i in annoy_ids if id_map[i] in self.kf_lookup]
                session_state['buffers'][index_name] = keyframes
                session_state['positions'][index_name] = buffer_size
        
        return session_state
    
    def _get_next_best_result(self, session_state: dict, filters: dict, query: str) -> 'Keyframe':
        best_keyframe = None
        best_score = float('inf')
        best_buffer_name = None
        
        for buffer_name, buffer in session_state['buffers'].items():
            if not buffer:
                continue
                
            candidate = buffer[0]
            if candidate.id in session_state['returned_ids']:
                buffer.pop(0)
                self._refill_buffer(buffer_name, session_state, query)
                continue
            
            score = self._compute_simple_score(candidate, query, buffer_name, filters)
            
            if score < best_score:
                best_score = score
                best_keyframe = candidate
                best_buffer_name = buffer_name
        
        if best_keyframe:
            session_state['buffers'][best_buffer_name].pop(0)
            self._refill_buffer(best_buffer_name, session_state, query)
        
        return best_keyframe
    
    def _refill_buffer(self, buffer_name: str, session_state: dict, query: str):
        current_pos = session_state['positions'][buffer_name]
        
        if buffer_name in ["transcript_text", "ocr_text", "dino", "colors", "objects", "ocr_embedding"]:
            return
        
        query_embedding = self.encode_text(query)
        
        if buffer_name == "clip":
            index, id_map = self.clip_index, self.id_map
        elif buffer_name == "colors":
            index, id_map = self.color_index, self.color_id_map
        elif buffer_name == "objects":
            index, id_map = self.object_index, self.object_id_map
        elif buffer_name == "ocr_embedding":
            index, id_map = self.ocr_index, self.ocr_id_map
        else:
            return
        
        try:
            next_batch_size = min(50, max(1, len(self.kf_lookup) - current_pos))
            if next_batch_size > 0:
                annoy_ids = index.get_nns_by_vector(query_embedding, current_pos + next_batch_size)
                if len(annoy_ids) > current_pos:
                    new_id = annoy_ids[current_pos]
                    if id_map[new_id] in self.kf_lookup:
                        session_state['buffers'][buffer_name].append(self.kf_lookup[id_map[new_id]])
                        session_state['positions'][buffer_name] += 1
        except (IndexError, KeyError):
            pass
    
    def _compute_simple_score(self, keyframe: 'Keyframe', query: str, buffer_name: str, filters: dict = None) -> float:
        base_score = 0.5
        
        if buffer_name == "transcript_text" and keyframe.transcript_text:
            query_words = set(query.lower().split())
            import re
            clean_transcript = re.sub(r'[^\w\s]', ' ', keyframe.transcript_text.lower())
            transcript_words = set(clean_transcript.split())
            
            if query_words.intersection(transcript_words):
                match_ratio = len(query_words.intersection(transcript_words)) / len(query_words)
                confidence_boost = max(0.3, keyframe.transcript_confidence or 0.3)
                base_score *= (0.3 - (0.2 * match_ratio * confidence_boost))
        
        if buffer_name == "ocr_text" and keyframe.ocr_text:
            query_words = set(query.lower().split())
            import re
            clean_ocr = re.sub(r'[^\w\s]', ' ', keyframe.ocr_text.lower())
            ocr_words = set(clean_ocr.split())
            
            if query_words.intersection(ocr_words):
                match_ratio = len(query_words.intersection(ocr_words)) / len(query_words)
                confidence_boost = max(0.3, keyframe.ocr_confidence or 0.3)
                base_score *= (0.2 - (0.15 * match_ratio * confidence_boost))
        
        if filters:
            davinci_penalty = self._compute_davinci_filter_penalty(keyframe, filters)
            base_score *= davinci_penalty
        
        return base_score
    
    def _compute_davinci_filter_penalty(self, keyframe: 'Keyframe', filters: dict) -> float:
        penalty = 1.0
        
        for kf_id, categories in filters.items():
            if not str(kf_id).startswith('davinci_'):
                continue
                
            davinci_features = self._get_davinci_features(str(kf_id))
            if not davinci_features:
                continue
                
            keyframe_features = keyframe.get_features_from_keyframe()
            
            for category in categories:
                similarity = self._compute_davinci_similarity(davinci_features, keyframe_features, category)
                penalty *= max(0.1, similarity)
        
        return penalty
    
    def _get_davinci_features(self, davinci_id: str) -> dict:
        from django.core.cache import cache
        return cache.get(f'davinci_features_{davinci_id}')
    
    def _compute_davinci_similarity(self, davinci_features: dict, keyframe_features: dict, category: str) -> float:
        if category == "embeddings":
            davinci_clip = davinci_features.get('clip_emb')
            davinci_dino = davinci_features.get('dino_emb')
            kf_clip = keyframe_features.get('clip_emb')
            kf_dino = keyframe_features.get('dino_emb')
            
            similarities = []
            if davinci_clip is not None and kf_clip is not None:
                clip_sim = max(0.0, np.dot(davinci_clip, kf_clip) / (np.linalg.norm(davinci_clip) * np.linalg.norm(kf_clip)))
                similarities.append(clip_sim)
            if davinci_dino is not None and kf_dino is not None:
                dino_sim = max(0.0, np.dot(davinci_dino, kf_dino) / (np.linalg.norm(davinci_dino) * np.linalg.norm(kf_dino)))
                similarities.append(dino_sim)
            
            return max(similarities) if similarities else 0.0
            
        elif category == "colors":
            davinci_hist = davinci_features.get('histogram')
            kf_hist = keyframe_features.get('histogram')
            
            if davinci_hist is not None and kf_hist is not None:
                return max(0.0, 1.0 - np.linalg.norm(davinci_hist - kf_hist))
            return 0.0
            
        elif category == "objects":
            davinci_obj = davinci_features.get('object_vector')
            kf_obj = keyframe_features.get('object_vector')
            
            if davinci_obj is not None and kf_obj is not None:
                return max(0.0, np.dot(davinci_obj, kf_obj) / (np.linalg.norm(davinci_obj) * np.linalg.norm(kf_obj)))
            return 0.0
            
        return 0.0

    def _search_transcripts(self, query: str, threshold: float = 0.3) -> set:
        matching_ids = set()
        query_words = set(query.lower().split())
        
        if not query_words:
            return matching_ids
            
        for kf_id, kf in self.kf_lookup.items():
            if not kf.transcript_text or not kf.transcript_text.strip():
                continue
                
            if kf.transcript_confidence is not None and kf.transcript_confidence < threshold:
                continue
                
            import re
            clean_transcript = re.sub(r'[^\w\s]', ' ', kf.transcript_text.lower())
            transcript_words = set(clean_transcript.split())
            if query_words.intersection(transcript_words):
                matching_ids.add(kf_id)
                continue
                
            if kf.transcript_context and kf.transcript_context.strip():
                clean_context = re.sub(r'[^\w\s]', ' ', kf.transcript_context.lower())
                context_words = set(clean_context.split())
                if query_words.intersection(context_words):
                    matching_ids.add(kf_id)
        
        return matching_ids

    def _search_ocr_text(self, query: str, threshold: float = 0.5) -> set:
        matching_ids = set()
        query_words = set(query.lower().split())
        
        if not query_words:
            return matching_ids
            
        for kf_id, kf in self.kf_lookup.items():
            if not kf.ocr_text or not kf.ocr_text.strip():
                continue
                
            if kf.ocr_confidence is not None and kf.ocr_confidence < threshold:
                continue
                
            import re
            clean_ocr = re.sub(r'[^\w\s]', ' ', kf.ocr_text.lower())
            ocr_words = set(clean_ocr.split())
            if query_words.intersection(ocr_words):
                matching_ids.add(kf_id)
        
        return matching_ids

    def encode_text(self, text: str) -> np.ndarray:
        if text == self.last_query and self.last_embedding is not None:
            return self.last_embedding

        try:
            inputs = self.tokenizer([text], return_tensors="pt")
            
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = inputs.to(self.device)
            
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
            
            features = features[0].cpu().numpy()
            features /= np.linalg.norm(features)
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.warning(f"CUDA error in encode_text, falling back to CPU: {e}")
                self.device = "cpu"
                self.model = self.model.cpu()
                inputs = self.tokenizer([text], return_tensors="pt")
                with torch.no_grad():
                    features = self.model.get_text_features(**inputs)
                features = features[0].cpu().numpy()
                features /= np.linalg.norm(features)
            else:
                raise

        self.last_query = text
        self.last_embedding = features

        query_objects = {
            "objects": ufil.find_fuzzy_object_matches(text, threshold=0.5, max_matches=5)
        }
        self.last_query_objects = query_objects

        return features

def prune_similar_results(results, clip_distance_threshold=0.05):
    selected = []
    seen_clips = set()

    for kf in results:
        if kf.clip.id in seen_clips:
            continue

        too_similar = False

        if not too_similar:
            selected.append(kf)
            seen_clips.add(kf.clip.id)

    return selected

def compute_adaptive_alpha(num_values: int, base_alpha: float = 5, max_alpha: float = 5.0, ramp : float = 1.3):
    return min(max_alpha, base_alpha + np.log1p(num_values - 1) * ramp)