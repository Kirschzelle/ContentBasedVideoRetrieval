from asyncio.windows_events import INFINITE
import time
import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPModel
from VideoSearch.models import Keyframe
from VideoSearch.utils.hardware import EmbeddingModelSelector
from VideoSearch.utils.visual_feature_extractor import compute_distance, nonlinear_pooling
import utils.filters as ufil
from utils.annoy_index import build_annoy_index

class Searcher:
    def __init__(self):
        hardware = EmbeddingModelSelector()
        clip_model_name, _, _ = hardware.select()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.model = CLIPModel.from_pretrained(clip_model_name).to(self.device)

        self.last_query = None
        self.last_embedding = None
        self.last_query_objects = None

        self.kf_lookup = {
            kf.id: kf for kf in Keyframe.objects.select_related("clip", "clip__video")
            .only("id", "clip__video__id", "clip__id", "frame",
                  "embedding_clip", "embedding_dino", "histogram_hsv",
                  "dominant_colors", "colorfulness", "object_vector")
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

    def search_incremental(self, query: str, returned_ids=None, filters=None, top_k=5):
        if returned_ids is None:
            returned_ids = set()
        if filters is None:
            filters = {}

        query_embedding = self.encode_text(query)

        clip_ids = self.clip_index.get_nns_by_vector(query_embedding, 5000)
        filter_ids = set()

        for kf, categories in filters.items():
            filter_kf = self.kf_lookup.get(kf)
            filter_feats = filter_kf.get_features_from_keyframe()

            for category in categories:
                if category == "embeddings":
                    emb = filter_feats.get("dino_emb")
                    if emb is not None:
                        dino_ids = self.dino_index.get_nns_by_vector(emb, 1000)
                        filter_ids.update(self.dino_id_map[i] for i in dino_ids)
                elif category == "colors":
                    hist = filter_feats.get("histogram")
                    if hist is not None:
                        color_ids = self.color_index.get_nns_by_vector(hist, 1000)
                        filter_ids.update(self.color_id_map[i] for i in color_ids)
                elif category == "objects":
                    obj_vec = filter_feats.get("object_vector")
                    if obj_vec is not None:
                        obj_ids = self.object_index.get_nns_by_vector(obj_vec, 1000)
                        filter_ids.update(self.object_id_map[i] for i in obj_ids)

        all_candidate_ids = set(clip_ids) | filter_ids
        all_candidate_ids.difference_update(returned_ids)

        scored = []
        for kf_id in all_candidate_ids:
            kf = self.kf_lookup.get(kf_id)
            if not kf:
                continue
            score = self.compute_total_similarity(query_embedding, kf, filters)
            if score is not None:
                scored.append((score, kf))

        scored.sort(key=lambda x: x[0])
        pruned = prune_similar_results([s[1] for s in scored])
        return pruned[:top_k]

    def compute_total_similarity(self, query_embedding, candidate_kf, filters):
        candidate_features = candidate_kf.get_features_from_keyframe()

        clip_score = self._compute_clip_similarity(query_embedding, candidate_features)
        if clip_score is None:
            return None

        object_score = self._compute_object_similarity(candidate_features)
        if object_score is None:
            return None

        filter_scores = self._compute_filter_distances(candidate_features, filters)

        distances = [clip_score] + filter_scores
        if object_score is not None:
            distances.insert(1, object_score)
        #alpha = compute_adaptive_alpha(len(distances))
        return nonlinear_pooling(distances, 1)

    def _compute_clip_similarity(self, query_embedding, candidate_features):
        emb = candidate_features.get("clip_emb")
        if emb is None:
            return None
        norm = np.linalg.norm(emb)
        if norm == 0:
            return None
        emb /= norm
        return 1 - np.dot(query_embedding, emb)

    def _compute_object_similarity(self, candidate_features):
        if self.last_query_objects is None:
            return None
        object_distance = ufil.filter_objects(candidate_features, self.last_query_objects)
        confs = self.last_query_objects.get("objects", {})
        avg_conf = np.mean(list(confs.values())) if confs else 0.0
        weight = 0.2 + 0.8 * avg_conf
        return object_distance

    def _compute_filter_distances(self, candidate_features, filters):
        distances = []
        for kf, categories in filters.items():
            filter_keyframe = Keyframe.objects.get(id=kf)
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
        #for sel in selected:
        #    dist = compute_distance(kf.get_features_from_keyframe(), sel.get_features_from_keyframe())
        #    if dist < clip_distance_threshold:
        #        too_similar = True
        #        break

        if not too_similar:
            selected.append(kf)
            seen_clips.add(kf.clip.id)

    return selected

def compute_adaptive_alpha(num_values: int, base_alpha: float = 5, max_alpha: float = 5.0, ramp : float = 1.3):
    return min(max_alpha, base_alpha + np.log1p(num_values - 1) * ramp)