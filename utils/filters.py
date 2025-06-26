from VideoSearch.utils.embeddings import calculate_combined_distance as filter_embedding_dist
from VideoSearch.utils.color_features import compute_distance as filter_colors_dist

def filter_embedding(canidate_features, filter_features):
    return filter_embedding_dist(canidate_features, filter_features)

def filter_colors(canidate_features, filter_features):
    return filter_colors_dist(canidate_features, filter_features)