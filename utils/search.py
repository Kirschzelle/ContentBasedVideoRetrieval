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
        self.last_search_mode = None

        self.kf_lookup = {
            kf.id: kf for kf in Keyframe.objects.select_related("clip", "clip__video")
            .only("id", "clip__video__id", "clip__id", "frame",
                  "embedding_clip", "embedding_dino", "histogram_hsv",
                  "dominant_colors", "colorfulness", "object_vector",
                  "transcript_text", "transcript_confidence", "transcript_context", "transcript_embedding",
                  "ocr_text", "ocr_confidence", "ocr_bboxes", "ocr_embedding")
        }


    def search_incremental(self, query: str, returned_ids=None, filters=None, search_mode="balanced", top_k=None):
        if returned_ids is None:
            returned_ids = set()
        if filters is None:
            filters = {}

        # Reset debug counter for each search
        self._debug_count = 0
        
        query_embedding = self.encode_text(query, search_mode)

        all_candidate_ids = set(self.kf_lookup.keys()) - returned_ids

        scored = []
        for kf_id in all_candidate_ids:
            kf = self.kf_lookup.get(kf_id)
            if not kf:
                continue
            score = self.compute_total_similarity(query_embedding, kf, filters, search_mode, query)
            if score is not None:
                scored.append((score, kf))

        scored.sort(key=lambda x: x[0])
        pruned = prune_similar_results([s[1] for s in scored])
        return pruned[:top_k] if top_k else pruned

    def compute_total_similarity(self, query_embedding, candidate_kf, filters, search_mode="balanced", query_text=None):
        candidate_features = candidate_kf.get_features_from_keyframe()

        clip_score = self._compute_clip_similarity(query_embedding, candidate_features)
        if clip_score is None:
            return None

        object_score = self._compute_object_similarity(candidate_features)
        ocr_score = self._compute_ocr_similarity(query_embedding, candidate_features, query_text)
        transcript_score = self._compute_transcript_similarity(query_embedding, candidate_features, query_text)
        
        
        filter_scores = self._compute_filter_distances(candidate_features, filters)

        distances = []
        weights = []
        
        if search_mode == "visual":
            distances.append(clip_score)
            weights.append(2.0)
            if object_score is not None:
                distances.append(object_score)
                weights.append(0.2)
            if ocr_score is not None:
                distances.append(ocr_score)
                weights.append(1.0)
        elif search_mode == "audio":
            distances.append(clip_score)
            weights.append(0.5)
            if ocr_score is not None:
                distances.append(ocr_score)
                weights.append(0.1)
            if transcript_score is not None:
                distances.append(transcript_score)
                weights.append(5.0)
        else:
            distances.append(clip_score)
            weights.append(1.0)
            if object_score is not None:
                distances.append(object_score)
                weights.append(0.1)
            if ocr_score is not None:
                distances.append(ocr_score)
                weights.append(1.0)
            if transcript_score is not None:
                distances.append(transcript_score)
                weights.append(1.0)
        
        for score in filter_scores:
            distances.append(score)
            weights.append(2.0)
        
        weighted_distances = [d * w for d, w in zip(distances, weights)]
        return nonlinear_pooling(weighted_distances, 1)
    
    def _compute_clip_similarity(self, query_embedding, candidate_features):
        emb = candidate_features.get("clip_emb")
        if emb is None:
            return None
        norm = np.linalg.norm(emb)
        if norm == 0:
            return None
        emb = emb / norm
        return 1 - np.dot(query_embedding, emb)

    def _compute_object_similarity(self, candidate_features):
        if self.last_query_objects is None:
            return None
        object_distance = ufil.filter_objects(candidate_features, self.last_query_objects)
        confs = self.last_query_objects.get("objects", {})
        avg_conf = np.mean(list(confs.values())) if confs else 0.0
        weight = 0.2 + 0.8 * avg_conf
        return object_distance * weight

    def _compute_ocr_similarity(self, query_embedding, candidate_features, query_text=None):
        ocr_data = candidate_features.get("ocr", {})
        ocr_text = ocr_data.get("text") if ocr_data else None
        ocr_emb = candidate_features.get("ocr_embedding")
        ocr_confidence = ocr_data.get("confidence", 0.0) if ocr_data else 0.0
        
        if not ocr_text and ocr_emb is None:
            return None
        
        scores = []
        weights = []
        
        if ocr_text and query_text:
            query_lower = query_text.lower()
            ocr_lower = ocr_text.lower()
            
            if query_lower in ocr_lower:
                exact_match_score = 0.0
            else:
                query_words = set(query_lower.split())
                ocr_words = set(ocr_lower.split())
                if query_words & ocr_words:
                    word_overlap = len(query_words & ocr_words) / len(query_words)
                    exact_match_score = 1.0 - word_overlap
                else:
                    exact_match_score = 1.0
            
            scores.append(exact_match_score)
            weights.append(ocr_confidence * 2.0)
        
        if ocr_emb is not None:
            norm = np.linalg.norm(ocr_emb)
            if norm > 0:
                ocr_emb = ocr_emb / norm
                if ocr_emb.shape == query_embedding.shape:
                    embedding_score = 1 - np.dot(query_embedding, ocr_emb)
                    scores.append(embedding_score)
                    weights.append(1.0)
        
        if not scores:
            return None
        
        if len(scores) == 1:
            return scores[0]
        
        weighted_avg = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        return weighted_avg

    def _compute_transcript_similarity(self, query_embedding, candidate_features, query_text=None):
        transcript_data = candidate_features.get("transcript", {})
        transcript_text = transcript_data.get("text") if transcript_data else None
        transcript_emb = candidate_features.get("transcript_embedding")
        transcript_confidence = transcript_data.get("confidence", 0.0) if transcript_data else 0.0
        
        if not transcript_text and transcript_emb is None:
            return None
        
        scores = []
        weights = []
        
        if transcript_text and query_text:
            query_lower = query_text.lower()
            transcript_lower = transcript_text.lower()
            
            if query_lower in transcript_lower:
                exact_match_score = 0.0
            else:
                query_words = set(query_lower.split())
                transcript_words = set(transcript_lower.split())
                if query_words & transcript_words:
                    word_overlap = len(query_words & transcript_words) / len(query_words)
                    exact_match_score = 1.0 - word_overlap
                else:
                    exact_match_score = 1.0
            
            scores.append(exact_match_score)
            weights.append(transcript_confidence * 2.0)
        
        if transcript_emb is not None:
            norm = np.linalg.norm(transcript_emb)
            if norm > 0:
                transcript_emb = transcript_emb / norm
                if transcript_emb.shape == query_embedding.shape:
                    embedding_score = 1 - np.dot(query_embedding, transcript_emb)
                    scores.append(embedding_score)
                    weights.append(1.0)
        
        if not scores:
            return None
        
        if len(scores) == 1:
            return scores[0]
        
        weighted_avg = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        return weighted_avg

    def _compute_filter_distances(self, candidate_features, filters):
        distances = []
        for kf_id, categories in filters.items():
            filter_keyframe = self.kf_lookup.get(kf_id)
            if not filter_keyframe:
                continue
            filter_features = filter_keyframe.get_features_from_keyframe()
            for category in categories:
                if category == "embeddings":
                    result = ufil.filter_embedding(candidate_features, filter_features)
                elif category == "colors":
                    result = ufil.filter_colors(candidate_features, filter_features)
                elif category == "objects":
                    result = ufil.filter_objects(candidate_features, filter_features)
                else:
                    continue
                distances.append(result)
        return distances


    def encode_text(self, text: str, search_mode: str = "balanced") -> np.ndarray:
        if text == self.last_query and search_mode == self.last_search_mode and self.last_embedding is not None:
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
                logger.warning(f"CUDA error in encode_text, reinitializing model on CPU: {e}")
                # Reinitialize model on CPU instead of moving existing model
                from VideoSearch.utils.hardware import EmbeddingModelSelector
                hardware = EmbeddingModelSelector()
                clip_model_name, _, _ = hardware.select()
                
                try:
                    self.model = CLIPModel.from_pretrained(clip_model_name, local_files_only=True)
                except (OSError, ValueError):
                    self.model = CLIPModel.from_pretrained(clip_model_name)
                
                self.device = "cpu"
                inputs = self.tokenizer([text], return_tensors="pt")
                with torch.no_grad():
                    features = self.model.get_text_features(**inputs)
                features = features[0].cpu().numpy()
                features /= np.linalg.norm(features)
            else:
                raise

        self.last_query = text
        self.last_search_mode = search_mode
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