from VideoSearch.utils.embeddings import calculate_combined_distance as filter_embedding_dist
from VideoSearch.utils.color_features import compute_distance as filter_colors_dist
from VideoSearch.utils.objects import soft_object_distance as filter_objects_dist
from VideoSearch.utils.objects import YOLO_CLASSES
import difflib

def filter_embedding(canidate_features, filter_features):
    return filter_embedding_dist(canidate_features, filter_features)

def filter_colors(canidate_features, filter_features):
    return filter_colors_dist(canidate_features, filter_features)

def filter_objects(canidate_features, filter_features):
    return filter_objects_dist(canidate_features, filter_features)

def find_fuzzy_object_matches(query: str, threshold=0.75, max_matches=5) -> dict:
    """
    Returns a dict mapping matched YOLO class names to similarity scores (0�1).
    """
    words = query.lower().split()
    matched = {}

    for word in words:
        best = difflib.get_close_matches(word, YOLO_CLASSES, n=1, cutoff=threshold)
        if best:
            match = best[0]
            ratio = difflib.SequenceMatcher(None, word, match).ratio()
            if ratio >= threshold and match not in matched:
                matched[match] = ratio

    return dict(sorted(matched.items(), key=lambda x: -x[1])[:max_matches])