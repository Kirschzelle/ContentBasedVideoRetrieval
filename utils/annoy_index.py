from annoy import AnnoyIndex
from VideoSearch.models import Keyframe
import numpy as np


def load_all_clip_embeddings():
    """Load and normalize all keyframe embeddings."""
    keyframes = Keyframe.objects.only("id", "embedding_clip")
    id_map = []
    embeddings = []

    for kf in keyframes:
        emb = Keyframe.decompress_array(kf.embedding_clip)
        emb /= np.linalg.norm(emb)
        id_map.append(kf.id)
        embeddings.append(emb.astype("float32"))

    return id_map, np.stack(embeddings)

def build_annoy_index(feature_name, kf_lookup, n_trees=700):
    dim = None
    index = None
    id_map = {}
    i = 0

    for kf_id, kf in kf_lookup.items():
        vec = kf.get_features_from_keyframe().get(feature_name)
        if vec is None or np.linalg.norm(vec) == 0:
            continue

        if dim is None:
            dim = len(vec)
            index = AnnoyIndex(dim, "angular")

        index.add_item(i, vec)
        id_map[i] = kf_id
        i += 1

    if i == 0 or index is None:
        raise ValueError(f"No valid vectors found for feature '{feature_name}'")

    index.build(n_trees)
    return index, id_map