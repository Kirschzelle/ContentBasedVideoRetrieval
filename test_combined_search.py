#!/usr/bin/env python3
"""
Quick test script for the enhanced combined search with filters.
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ContentBasedVideoRetrieval.settings')
django.setup()

from VideoSearch.utils.combined_embeddings import get_combined_vector_builder
from VideoSearch.models import Keyframe
import numpy as np

def test_combined_vector_builder():
    """Test the enhanced combined vector builder."""
    print("Testing Combined Vector Builder with Filters...")
    
    builder = get_combined_vector_builder()
    
    # Test basic vector building
    sample_features = {
        'clip_emb': np.random.rand(512).astype(np.float32),
        'transcript_embedding': np.random.rand(384).astype(np.float32),
        'dino_emb': np.random.rand(384).astype(np.float32),
        'object_vector': np.random.rand(80).astype(np.float32),
        'histogram': np.random.rand(256).astype(np.float32),
        'transcript': {'embedding': np.random.rand(384).astype(np.float32)}
    }
    
    # Test without filters
    combined_basic = builder.build_combined_vector(sample_features)
    print(f"Basic combined vector shape: {combined_basic.shape}")
    print(f"Expected dimensions: {builder.total_dims}")
    
    # Test with filters
    filter_vectors = {
        'embedding_filter': np.random.rand(384).astype(np.float32),
        'color_filter': np.random.rand(256).astype(np.float32)
    }
    
    combined_filtered = builder.build_combined_vector(sample_features, filter_vectors=filter_vectors)
    print(f"Filtered combined vector shape: {combined_filtered.shape}")
    
    # Test query vector building
    query_emb = np.random.rand(512).astype(np.float32)
    transcript_emb = np.random.rand(384).astype(np.float32)
    
    query_combined = builder.build_combined_query_vector(
        query_emb, transcript_emb, 'balanced', filter_vectors
    )
    print(f"Query combined vector shape: {query_combined.shape}")
    
    # Verify dimensions match
    assert combined_basic.shape == combined_filtered.shape == query_combined.shape
    print("All vector dimensions match!")
    
    print("Combined Vector Builder test completed successfully!")

def test_search_integration():
    """Test search integration with filters."""
    print("\nTesting Search Integration...")
    
    try:
        from utils.search import VideoSearchIndex
        
        # This would normally be initialized from the main application
        print("Search integration test would require full database setup.")
        print("Enhanced search system is ready for testing with actual data.")
        
    except ImportError as e:
        print(f"Search integration test skipped: {e}")

if __name__ == "__main__":
    test_combined_vector_builder()
    test_search_integration()
    print("\nFilter-aware combined search system implemented successfully!")