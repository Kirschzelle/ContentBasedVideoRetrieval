import numpy as np
from VideoSearch.utils.embeddings import ImageEmbedder, calculate_combined_distance as embedding_distance, get_distance_to_existing_keyframes as embedding_ex_keyframes_distance
from VideoSearch.utils.color_features import ColorFeatureExtractor , compute_distance as color_distance, distance_to_existing_keyframes as color_ex_keyframes_distance
import time
from VideoSearch.utils.objects import soft_object_distance as object_distance

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

    def extract_features_batch(self, images):
        """
        Batched version of extract_features. Returns a list of feature dicts.
        """
        all_features = [{} for _ in images]

        t0 = time.perf_counter()

        if self.use_embeddings:
            embeddings = self.embedder.get_combined_embedding_batch(images)
            for i in range(len(images)):
                all_features[i].update(embeddings[i])

        t1 = time.perf_counter()

        if self.use_color:
            colors = self.color.extract_all_batch(images)
            for i in range(len(images)):
                all_features[i].update(colors[i])

        t2 = time.perf_counter()

        if self.command:
            self.command.stdout.write(self.command.style_info(
                f"Embedding batch: {(t1 - t0) * 1000:.1f}ms | Color batch: {(t2 - t1) * 1000:.1f}ms"
            ))

        return all_features

    def get_candidates(self, images, start, step_size, clip, threshold):
        """
        Returns a list of (frame_number, features_dict) tuples for frames
        that are sufficiently different from existing keyframes.
        Uses batched feature extraction for performance.
        """
        indices = [i for i in range(len(images)) if i == 0 or i % step_size == 0]
        selected_images = [images[i] for i in indices]
        selected_frame_numbers = [start + i for i in indices]

        batched_features = self.extract_features_batch(selected_images)

        candidates = []
        for frame_number, features in zip(selected_frame_numbers, batched_features):
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

        if len(candidates) == 1:
            return candidates[0]

        for i, (frame_i, feat_i) in enumerate(candidates):
            distances = []

            for j, (_, feat_j) in enumerate(candidates):
                if i == j:
                    continue

                d = compute_distance(feat_i, feat_j)
                distances.append(d)

            if distances:
                median_dist = np.median(distances)
                if median_dist < best_score:
                    best_score = median_dist
                    best_frame = (frame_i, feat_i)

        return best_frame
    
    def distance_to_existing_keyframes(self, clip, features):
        min_distances = []
        max_distances = []

        if self.use_embeddings:
            min_dist_emb, max_dist_emb = embedding_ex_keyframes_distance(clip, features)
            min_distances.append(min_dist_emb)
            max_distances.append(max_dist_emb)

        if self.use_color:
            min_dist_col, max_dist_col = color_ex_keyframes_distance(clip, features)
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
                min_dist, _ = self.distance_to_existing_keyframes(
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

def compute_distance(feat_a, feat_b):
    """
    Compute similarity distance between two feature dicts.
    Currently uses embeddings + color distances.
    """
    distances = []
        
    if "clip_emb" in feat_a and "clip_emb" in feat_b:
        dist = embedding_distance(feat_a, feat_b)
        distances.append(dist)

    if "histogram" in feat_a and "histogram" in feat_b:
        dist = color_distance(feat_a, feat_b)
        distances.append(dist)

    if "object_vector" in feat_a and "object_vector" in feat_b:
        dist = object_distance(feat_a, feat_b)
        distances.append(dist)

    combined_score = nonlinear_pooling(distances)

    if not distances:
        return 1.0

    return combined_score