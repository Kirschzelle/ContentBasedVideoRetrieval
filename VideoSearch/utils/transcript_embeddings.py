import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class TranscriptEmbedder:
    """Transcript text embeddings using sentence transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize transcript embedder.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self.model = None
        self._embedding_dim = None
    
    def _load_model(self):
        """Load sentence transformer model on first use."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                self._embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"Loaded transcript embedding model: {self.model_name} (dim: {self._embedding_dim})")
            except ImportError:
                logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
                raise ImportError("sentence-transformers package required for transcript embeddings")
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim
    
    def encode_text(self, text: str, normalize: bool = True) -> Optional[np.ndarray]:
        """
        Encode text to embedding vector.
        
        Args:
            text: Input text to encode
            normalize: Whether to L2 normalize the embedding
            
        Returns:
            Embedding vector or None if text is empty/None
        """
        if not text or not text.strip():
            return None
            
        self._load_model()
        
        try:
            embedding = self.model.encode(text.strip())
            
            if normalize and np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
                
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error encoding text '{text[:100]}...': {e}")
            return None
    
    def encode_texts(self, texts: List[str], normalize: bool = True) -> List[Optional[np.ndarray]]:
        """
        Encode multiple texts to embeddings.
        
        Args:
            texts: List of texts to encode
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            List of embedding vectors (None for empty texts)
        """
        if not texts:
            return []
            
        self._load_model()
        
        # Filter out empty texts but track indices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text.strip())
                valid_indices.append(i)
        
        if not valid_texts:
            return [None] * len(texts)
        
        try:
            # Batch encode valid texts
            embeddings = self.model.encode(valid_texts)
            
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = np.where(norms > 0, embeddings / norms, embeddings)
            
            # Map back to original indices
            result = [None] * len(texts)
            for i, embedding in enumerate(embeddings):
                original_idx = valid_indices[i]
                result[original_idx] = embedding.astype(np.float32)
                
            return result
            
        except Exception as e:
            logger.error(f"Error encoding {len(valid_texts)} texts: {e}")
            return [None] * len(texts)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0-1)
        """
        emb1 = self.encode_text(text1)
        emb2 = self.encode_text(text2)
        
        if emb1 is None or emb2 is None:
            return 0.0
            
        return max(0.0, np.dot(emb1, emb2))  # Cosine similarity (already normalized)


def get_transcript_embedder() -> TranscriptEmbedder:
    """Get shared transcript embedder instance."""
    if not hasattr(get_transcript_embedder, '_instance'):
        get_transcript_embedder._instance = TranscriptEmbedder()
    return get_transcript_embedder._instance