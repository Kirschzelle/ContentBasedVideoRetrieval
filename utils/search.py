import time
import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPModel
from VideoSearch.models import Keyframe
from VideoSearch.utils.hardware import EmbeddingModelSelector
from VideoSearch.utils.visual_feature_extractor import compute_distance, nonlinear_pooling
import utils.filters as ufil

class Searcher:
    def __init__(self):
        hardware = EmbeddingModelSelector()
        clip_model_name, dino_model_name, mode = hardware.select()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.model = CLIPModel.from_pretrained(clip_model_name).to(self.device)

        self.last_query = None
        self.last_embedding = None
        self.last_query_objects = None
        self.remaining_keyframes = []
        self.remaining_size = -1

    def search_incremental(self, query: str, returned_ids=None, filters=None, top_k=1):
        if returned_ids is None:
            returned_ids = set()
        if filters is None:
            filters = {}

        query_embedding = self.encode_text(query)

        if self.remaining_size != len(returned_ids):
            self.remaining_size = 0
            self.remaining_keyframes = list(
                Keyframe.objects.select_related("clip", "clip__video").all()
            )

        best_match = None
        best_similarity = -1.0

        for idx, kf in enumerate(self.remaining_keyframes):
            score = self.compute_total_similarity(query_embedding, kf, filters)
            if score is None:
                continue
            if score > best_similarity:
                best_similarity = score
                best_match = kf
                match_index = idx

        if best_match:
            self.remaining_size += 1
            filter_count = sum(len(v) for v in filters.values())
            base_threshold = 0.45
            reduction_per_filter = 0.1
            adaptive_threshold = max(0.0, base_threshold - filter_count * reduction_per_filter)

            self.remove_similar_keyframes(best_match, match_index, adaptive_threshold)
            return [best_match]

        return []

    def compute_total_similarity(self, query_embedding, candidate_kf, filters):
        candidate_features = candidate_kf.get_features_from_keyframe()

        clip_score = self._compute_clip_similarity(query_embedding, candidate_features)
        if clip_score is None:
            return None

        object_score = self._compute_object_similarity(candidate_features)
        if object_score is None:
            return None

        filter_scores = self._compute_filter_distances(candidate_features, filters)

        distances = [clip_score, object_score] + filter_scores
        alpha = compute_adaptive_alpha(len(distances))
        return nonlinear_pooling(distances, alpha)


    def _compute_clip_similarity(self, query_embedding, candidate_features):
        emb = candidate_features.get("clip_emb")
        if emb is None:
            return None
        norm = np.linalg.norm(emb)
        if norm == 0:
            return None
        emb /= norm
        return np.dot(query_embedding, emb)


    def _compute_object_similarity(self, candidate_features):
        if self.last_query_objects is None:
            return None
        object_distance = ufil.soft_object_distance(candidate_features, self.last_query_objects)
        confs = self.last_query_objects.get("objects", {})
        avg_conf = np.mean(list(confs.values())) if confs else 0.0
        weight = 0.2 + 0.8 * avg_conf
        return object_distance * weight


    def _compute_filter_distances(self, candidate_features, filters):
        distances = []
        for kf, categories in filters.items():
            filter_features = kf.get_features_from_keyframe()
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

    def encode_text(self, text: str) -> np.ndarray:
        if text == self.last_query and self.last_embedding is not None:
            return self.last_embedding

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        features = features[0].cpu().numpy()
        features /= np.linalg.norm(features)

        self.last_query = text
        self.last_embedding = features

        query_objects = {
            "objects": ufil.find_fuzzy_object_matches(text, threshold=0.1, max_matches=50)
        }
        self.last_query_objects = query_objects

        return features

    def remove_similar_keyframes(self, keyframe, match_index, threshold = 0.05):
        del self.remaining_keyframes[match_index]

        to_remove = []
        for idx, kf in enumerate(self.remaining_keyframes):
            if kf.clip.id == keyframe.clip.id:
                to_remove.append(idx)
                continue
            distance = compute_distance(keyframe.get_features_from_keyframe(), kf.get_features_from_keyframe())
            if distance < threshold:
                to_remove.append(idx)

        for idx in reversed(to_remove):
            del self.remaining_keyframes[idx]

def compute_adaptive_alpha(num_values: int, base_alpha: float = 0.5, max_alpha: float = 7.0, ramp : float = 1.5):
    return min(max_alpha, base_alpha + np.log1p(num_values - 1) * ramp)