import time
import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPModel
from VideoSearch.models import Keyframe
from VideoSearch.utils.hardware import EmbeddingModelSelector
from VideoSearch.utils.visual_feature_extractor import compute_distance

class Searcher:
    def __init__(self):
        hardware = EmbeddingModelSelector()
        clip_model_name, dino_model_name, mode = hardware.select()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.model = CLIPModel.from_pretrained(clip_model_name).to(self.device)

        self.last_query = None
        self.last_embedding = None
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
            emb = kf.load_embedding_clip()
            if emb is None:
                continue

            emb /= np.linalg.norm(emb)
            similarity = np.dot(query_embedding, emb)

            if similarity > best_similarity:
                best_similarity = similarity
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