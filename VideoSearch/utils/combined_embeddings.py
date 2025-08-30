import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class CombinedVectorBuilder:
    """Build combined vectors from multiple modalities with configurable weighting."""
    
    def __init__(self):
        """Initialize with default target dimensions and weights."""
        self.target_dims = {
            'clip_emb': 512,           # CLIP embedding (base dimension)
            'transcript_embedding': 512,  # Transcript embedding (repeated from 384)
            'dino_emb': 256,           # DINO embedding (reduced from original)
            'object_vector': 128,      # Object vector (repeated from 80)
            'histogram': 128,          # Color histogram (reduced)
        }
        
        self.weights = {
            'clip_emb': 1.0,           # Primary visual content
            'transcript_embedding': 0.8,  # Important semantic info
            'dino_emb': 0.6,           # Visual similarity
            'object_vector': 0.4,      # Object content
            'histogram': 0.3,          # Color matching
        }
        
        # Reserve space for filter vectors (128 dims each, up to 4 types)
        self.filter_dim = 128
        self.max_filters = 4  # embeddings, colors, objects, transcripts
        self.filter_space = self.filter_dim * self.max_filters  # 512 dims for filters
        
        self.total_dims = sum(self.target_dims.values()) + self.filter_space  # ~2048 total
        
    def _resize_vector(self, vector: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Resize vector to target dimension by repeating or truncating.
        
        Args:
            vector: Input vector
            target_dim: Target dimension
            
        Returns:
            Resized vector
        """
        if vector is None or len(vector) == 0:
            return np.zeros(target_dim, dtype=np.float32)
            
        current_dim = len(vector)
        
        if current_dim == target_dim:
            return vector.astype(np.float32)
        elif current_dim < target_dim:
            # Repeat vector to reach target dimension
            repeat_count = target_dim // current_dim
            remainder = target_dim % current_dim
            
            repeated = np.tile(vector, repeat_count)
            if remainder > 0:
                repeated = np.concatenate([repeated, vector[:remainder]])
            return repeated.astype(np.float32)
        else:
            # Truncate to target dimension
            return vector[:target_dim].astype(np.float32)
    
    def build_combined_vector(self, features: Dict, normalize: bool = True, filter_vectors: Dict = None) -> np.ndarray:
        """
        Build combined vector from keyframe features with optional filter vectors.
        
        Args:
            features: Dictionary of keyframe features
            normalize: Whether to normalize final vector
            filter_vectors: Optional dict of filter vectors to append
            
        Returns:
            Combined vector with optional filter components
        """
        components = []
        
        for feature_name, target_dim in self.target_dims.items():
            # Get feature vector
            if feature_name == 'transcript_embedding':
                # Special handling for transcript embeddings (nested in transcript dict)
                transcript_data = features.get('transcript', {})
                vector = transcript_data.get('embedding') if transcript_data else None
            else:
                vector = features.get(feature_name)
            
            # Resize to target dimension
            resized = self._resize_vector(vector, target_dim)
            
            # Apply weight
            weight = self.weights.get(feature_name, 1.0)
            weighted = resized * weight
            
            components.append(weighted)
        
        # Add filter vectors in reserved slots (always same total dimension)
        filter_types = ["embedding_filter", "color_filter", "object_filter", "transcript_filter"]
        for filter_type in filter_types:
            if filter_vectors and filter_type in filter_vectors:
                filter_vector = filter_vectors[filter_type]
                filter_resized = self._resize_vector(filter_vector, self.filter_dim)
            else:
                # Use zero vector for unused filter slots
                filter_resized = np.zeros(self.filter_dim, dtype=np.float32)
            components.append(filter_resized)
        
        # Concatenate all components
        combined = np.concatenate(components)
        
        # Normalize final vector
        if normalize:
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
        
        return combined.astype(np.float32)
    
    def build_combined_query_vector(self, 
                                  query_embedding: np.ndarray,
                                  transcript_embedding: Optional[np.ndarray] = None,
                                  query_type: str = 'balanced',
                                  filter_vectors: Dict = None) -> np.ndarray:
        """
        Build combined query vector for search with optional filter vectors.
        
        Args:
            query_embedding: CLIP text encoding of query
            transcript_embedding: Optional transcript encoding
            query_type: 'visual', 'text', 'balanced'
            filter_vectors: Optional dict of filter vectors to append
            
        Returns:
            Combined query vector with optional filter components
        """
        # Adjust weights based on query type
        if query_type == 'visual':
            weights = {**self.weights, 'clip_emb': 1.5, 'dino_emb': 1.0, 'transcript_embedding': 0.3}
        elif query_type == 'text':
            weights = {**self.weights, 'transcript_embedding': 1.2, 'clip_emb': 0.8}
        else:  # balanced
            weights = self.weights
            
        # Build query features dict
        query_features = {
            'clip_emb': query_embedding,
            'transcript_embedding': transcript_embedding,
            'dino_emb': None,  # Not available for queries
            'object_vector': None,  # Not available for queries  
            'histogram': None,  # Not available for queries
        }
        
        # Temporarily override weights
        original_weights = self.weights
        self.weights = weights
        
        try:
            combined = self.build_combined_vector(
                {'transcript': {'embedding': transcript_embedding}, **query_features}, 
                filter_vectors=filter_vectors
            )
            return combined
        finally:
            # Restore original weights
            self.weights = original_weights
    
    def get_dimensions(self) -> Dict[str, int]:
        """Get dimension breakdown."""
        return {
            'individual': dict(self.target_dims),
            'total': self.total_dims,
            'weights': dict(self.weights)
        }

def get_combined_vector_builder() -> CombinedVectorBuilder:
    """Get shared combined vector builder instance."""
    if not hasattr(get_combined_vector_builder, '_instance'):
        get_combined_vector_builder._instance = CombinedVectorBuilder()
    return get_combined_vector_builder._instance