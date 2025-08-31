"""
Process external frames (from DaVinci, etc.) for use in search filters.
"""

import tempfile
import os
from pathlib import Path
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def process_external_frame_for_search(uploaded_file):
    """
    Process an uploaded frame and extract all features for search filtering.
    
    Args:
        uploaded_file: Django UploadedFile object (image from DaVinci export)
        
    Returns:
        Dict with extracted features that can be used as search filters
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name
        
        try:
            # Load image
            image = Image.open(temp_path).convert('RGB')
            image_array = np.array(image)
            
            # Extract all features using existing pipeline
            features = extract_features_from_image(image, image_array)
            
            logger.info(f"Extracted features from external frame: {uploaded_file.name}")
            
            return {
                'success': True,
                'features': features,
                'image_info': {
                    'filename': uploaded_file.name,
                    'size': uploaded_file.size,
                    'dimensions': f"{image.width}x{image.height}"
                }
            }
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Failed to process external frame: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def extract_features_from_image(pil_image, image_array):
    """
    Extract all features from a single image using existing feature extraction pipeline.
    
    Args:
        pil_image: PIL Image object
        image_array: numpy array of image
        
    Returns:
        Dict with all extracted features
    """
    features = {}
    
    try:
        # CLIP embeddings
        features['clip_emb'] = extract_clip_embedding(pil_image)
        
        # DINO embeddings  
        features['dino_emb'] = extract_dino_embedding(pil_image)
        
        # Color features
        features['histogram'] = extract_color_histogram(image_array)
        features['dominant_colors'] = extract_dominant_colors(image_array)
        features['colorfulness'] = calculate_colorfulness(image_array)
        
        # Object detection
        features['object_vector'] = extract_object_features(pil_image)
        
        logger.info("Successfully extracted all features from image")
        
    except Exception as e:
        logger.error(f"Feature extraction error: {str(e)}")
        # Continue with partial features - don't fail completely
    
    return features

def extract_clip_embedding(pil_image):
    """Extract CLIP embedding from image."""
    try:
        # Use existing CLIP model - need to import from your existing code
        from utils.feature_extraction import get_clip_model  # Adjust import path
        
        model, preprocess = get_clip_model()
        
        # Preprocess and encode
        image_tensor = preprocess(pil_image).unsqueeze(0)
        
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            embedding = image_features.cpu().numpy().flatten()
            
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.astype(np.float32)
        
    except Exception as e:
        logger.warning(f"CLIP extraction failed: {e}")
        return None

def extract_dino_embedding(pil_image):
    """Extract DINO embedding from image.""" 
    try:
        # Use existing DINO model - need to import from your existing code
        from utils.feature_extraction import get_dino_model  # Adjust import path
        
        model, transform = get_dino_model()
        
        # Transform and encode
        image_tensor = transform(pil_image).unsqueeze(0)
        
        with torch.no_grad():
            features = model(image_tensor)
            embedding = features.cpu().numpy().flatten()
            
        # Normalize  
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.astype(np.float32)
        
    except Exception as e:
        logger.warning(f"DINO extraction failed: {e}")
        return None

def extract_color_histogram(image_array):
    """Extract HSV color histogram."""
    try:
        import cv2
        
        # Convert to HSV
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # Calculate histogram
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        
        # Normalize and flatten
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist.astype(np.float32)
        
    except Exception as e:
        logger.warning(f"Color histogram extraction failed: {e}")
        return None

def extract_dominant_colors(image_array, n_colors=5):
    """Extract dominant colors using k-means."""
    try:
        from sklearn.cluster import KMeans
        
        # Reshape for k-means
        pixels = image_array.reshape(-1, 3)
        
        # Reduce data size for performance
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get dominant colors
        colors = kmeans.cluster_centers_
        
        return colors.flatten().astype(np.float32)
        
    except Exception as e:
        logger.warning(f"Dominant colors extraction failed: {e}")
        return None

def calculate_colorfulness(image_array):
    """Calculate colorfulness metric.""" 
    try:
        # Split channels
        r, g, b = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]
        
        # Calculate RG and YB
        rg = np.absolute(r - g)
        yb = np.absolute(0.5 * (r + g) - b)
        
        # Calculate std and mean
        rb_std, rb_mean = np.std(rg), np.mean(rg)
        yb_std, yb_mean = np.std(yb), np.mean(yb)
        
        # Colorfulness metric
        colorfulness = np.sqrt(rb_std**2 + yb_std**2) + 0.3 * np.sqrt(rb_mean**2 + yb_mean**2)
        
        return float(colorfulness)
        
    except Exception as e:
        logger.warning(f"Colorfulness calculation failed: {e}")
        return None

def extract_object_features(pil_image):
    """Extract object detection features."""
    try:
        # Use existing object detection - need to import from your existing code
        from utils.feature_extraction import get_object_detector  # Adjust import path
        
        detector = get_object_detector()
        
        # Run detection
        results = detector(pil_image)
        
        # Convert to feature vector (same format as existing keyframes)
        object_vector = convert_detection_to_vector(results)
        
        return object_vector.astype(np.float32)
        
    except Exception as e:
        logger.warning(f"Object detection failed: {e}")
        return None

def convert_detection_to_vector(detection_results):
    """Convert object detection results to feature vector."""
    # This should match your existing object vector format
    # Placeholder implementation - adjust based on your existing code
    try:
        # Create 80-dimensional vector (COCO classes)
        vector = np.zeros(80, dtype=np.float32)
        
        # Fill with detection scores/counts
        # (Implementation depends on your existing object detection format)
        
        return vector
        
    except Exception as e:
        logger.warning(f"Object vector conversion failed: {e}")
        return np.zeros(80, dtype=np.float32)