import numpy as np
from VideoSearch.utils.embeddings import ImageEmbedder
from VideoSearch.utils.color_features import ColorFeatureExtractor

class VisualFeatureExtractor:
    def __init__(self, use_embeddings=True, use_color=True, command=None):
        self.use_embeddings = use_embeddings
        self.use_color = use_color
        self.command = command
        if use_embeddings:
            self.embedder = ImageEmbedder(command=command)
        if use_color:
            self.color = ColorFeatureExtractor(command=command)

    def extract_features(self, image):
        features = {}

        if self.use_embeddings:
            features.update(self.embedder.get_combined_embedding(image))

        if self.use_color:
            features.update(self.color.extract_all(image))

        return features

    def get_candidates(self, images, start, step_size, clip, threshold):
        """
        Returns a list of (frame_number, features_dict) tuples for frames
        that are sufficiently different from existing keyframes.
        """
        candidates = []

        for i, image in enumerate(images):
            if i != 0 and i % step_size != 0:
                continue

            frame_number = start + i
            features = self.extract_features(image)

            distance, _ = self.distance_to_existing_keyframes(clip, features)

            if distance >= threshold:
                candidates.append((frame_number, features))

        return candidates

    def select_representative(self, candidates):
        """
        From a list of (frame_number, features_dict), pick the most representative one
        based on median distance to others.
        """
        best_frame = None
        best_score = float("inf")

        for i, (frame_i, feat_i) in enumerate(candidates):
            distances = []

            for j, (_, feat_j) in enumerate(candidates):
                if i == j:
                    continue

                d = self.compute_distance(feat_i, feat_j)
                distances.append(d)

            if distances:
                median_dist = np.median(distances)
                if median_dist < best_score:
                    best_score = median_dist
                    best_frame = (frame_i, feat_i)

        return best_frame

    def compute_distance(self, feat_a, feat_b):
        """
        Compute similarity distance between two feature dicts.
        Currently uses embeddings + color distances.
        """
        distances = []
        
        if self.use_embeddings:
            dist = self.embedder.calculate_combined_distance(feat_a, feat_b)
            distances.append(dist)

        if self.use_color:
            dist = self.color.compute_distance(feat_a, feat_b)
            distances.append(dist)

        combined_score = nonlinear_pooling(distances)

        if not distances:
            return 1.0

        return combined_score
    
    def distance_to_existing_keyframes(self, clip, features):
        min_distances = []
        max_distances = []

        if self.use_embeddings:
            min_dist_emb, max_dist_emb = self.embedder.get_distance_to_existing_keyframes(clip, features)
            min_distances.append(min_dist_emb)
            max_distances.append(max_dist_emb)

        if self.use_color:
            min_dist_col, max_dist_col = self.color.distance_to_existing_keyframes(clip, features)
            min_distances.append(min_dist_col)
            max_distances.append(max_dist_col)

        return nonlinear_pooling(min_distances), nonlinear_pooling(max_distances)

    def filter_against_existing(self, clip, candidates, threshold):
        """
        Filters out candidates that are too similar to existing keyframes.
        """
        filtered = []
        for frame_number, features in candidates:
            if self.use_embeddings:
                min_dist, _ = self.embedder.get_distance_to_existing_keyframes(
                    clip,
                    features
                )
                if min_dist <= threshold:
                    continue
            filtered.append((frame_number, features))
        return filtered

def nonlinear_pooling(distances: list[float], alpha: float = 5.0) -> float:
    """
    Combines multiple distance values into a single score ∈ [0, 1],
    emphasizing high distances more strongly (softmax-style).
    """
    if not distances:
        return 1.0
    distances = np.array(distances)
    weights = np.exp(alpha * distances)
    return float(np.sum(distances * weights) / np.sum(weights))