import numpy as np
from typing import Dict, List, Optional
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from django.utils import timezone
from datetime import timedelta
import json

logger = logging.getLogger(__name__)

class SmartWeightingSystem:
    """
    Smart weighting system that learns optimal weights from user interaction data.
    
    Uses machine learning on click-through data to adjust feature weights in the
    combined vector builder for better search relevance.
    """
    
    def __init__(self, min_training_samples: int = 100):
        self.min_training_samples = min_training_samples
        self.scalers = {}  # Per query-type feature scalers
        self.models = {}   # Per query-type ML models
        
    def collect_search_interaction(self, query: str, query_type: str, 
                                 results: List, filters: Dict = None,
                                 user=None, session_id: str = None) -> int:
        """
        Record a search interaction and initialize engagement tracking.
        
        Returns:
            SearchInteraction ID for tracking engagement
        """
        from VideoSearch.models import SearchInteraction, ResultEngagement
        
        filter_types = list(filters.keys()) if filters else []
        keyframe_ids = [kf.id for kf in results]
        
        interaction = SearchInteraction.objects.create(
            query=query,
            query_type=query_type,
            session_id=session_id,
            had_filters=bool(filters),
            filter_types=json.dumps(filter_types),
            total_results_returned=len(results),
            results_keyframe_ids=json.dumps(keyframe_ids)
        )
        
        # Initialize engagement tracking for all results
        for position, keyframe in enumerate(results):
            ResultEngagement.objects.create(
                search_interaction=interaction,
                keyframe=keyframe,
                result_position=position
            )
        
        return interaction.id
    
    def update_result_engagement(self, interaction_id: int, keyframe_id: int, 
                               engagement_data: dict):
        """
        Update engagement data for a specific result.
        
        Args:
            interaction_id: Search interaction ID
            keyframe_id: Keyframe that was engaged with
            engagement_data: Dict with engagement metrics like:
                - was_clicked: bool
                - click_timestamp: datetime  
                - time_to_click: float
                - view_duration: float
                - hover_duration: float
                - video_played: bool
                - video_watch_duration: float
                - video_completion_rate: float
                - was_downloaded: bool
                - was_shared: bool
                - was_bookmarked: bool
                - was_used_as_filter: bool
                - next_action: str
        """
        from VideoSearch.models import ResultEngagement, SearchInteraction
        
        try:
            engagement = ResultEngagement.objects.get(
                search_interaction_id=interaction_id,
                keyframe_id=keyframe_id
            )
            
            # Update all provided engagement fields
            for field, value in engagement_data.items():
                if hasattr(engagement, field) and value is not None:
                    setattr(engagement, field, value)
            
            # Save will automatically recalculate engagement_score
            engagement.save()
            
            logger.debug(f"Updated engagement for keyframe {keyframe_id}: score={engagement.engagement_score:.3f}")
            
        except ResultEngagement.DoesNotExist:
            logger.warning(f"Engagement record not found: interaction={interaction_id}, keyframe={keyframe_id}")
        except Exception as e:
            logger.error(f"Failed to update engagement: {e}")
    
    def record_result_click(self, interaction_id: int, clicked_keyframe_id: int,
                           result_position: int, time_to_click: float,
                           view_duration: float = None, next_action: str = None):
        """Legacy method - use update_result_engagement for richer tracking."""
        engagement_data = {
            'was_clicked': True,
            'click_timestamp': timezone.now(),
            'time_to_click': time_to_click,
            'view_duration': view_duration or 0.0,
            'next_action': next_action
        }
        self.update_result_engagement(interaction_id, clicked_keyframe_id, engagement_data)
    
    def prepare_training_data(self, query_type: str, days_back: int = 30, 
                            min_engagement_threshold: float = 0.1):
        """
        Prepare training data from recent user engagement.
        
        Args:
            query_type: Query type to train for
            days_back: How many days of data to include
            min_engagement_threshold: Minimum engagement score to include in training
            
        Returns:
            (features, labels, weights) where:
            - features: Feature vectors for each result
            - labels: Engagement quality (0.0-1.0, can be used for regression)  
            - weights: Sample weights based on engagement reliability
        """
        from VideoSearch.models import SearchInteraction, ResultEngagement, Keyframe
        
        # Get recent interactions for this query type
        cutoff_date = timezone.now() - timedelta(days=days_back)
        interactions = SearchInteraction.objects.filter(
            query_type=query_type,
            search_timestamp__gte=cutoff_date
        ).prefetch_related('resultengagement_set')
        
        features = []
        labels = []  # Now continuous engagement scores instead of binary
        sample_weights = []
        
        for interaction in interactions:
            for engagement in interaction.resultengagement_set.all():
                try:
                    keyframe = Keyframe.objects.get(id=engagement.keyframe_id)
                    kf_features = keyframe.get_features_from_keyframe()
                    
                    # Extract feature similarities 
                    feature_vector = self._extract_feature_similarities(
                        interaction.query, kf_features, interaction.query_type
                    )
                    
                    # Add position bias feature (earlier results get more attention)
                    position_bias = 1.0 / (engagement.result_position + 1)
                    feature_vector.append(position_bias)
                    
                    # Add interaction context features
                    feature_vector.extend([
                        1.0 if interaction.had_filters else 0.0,  # Had filters
                        float(interaction.total_results_returned) / 100.0,  # Result set size (normalized)
                    ])
                    
                    features.append(feature_vector)
                    labels.append(engagement.engagement_score)
                    
                    # Sample weighting: higher weights for more reliable engagement signals
                    weight = self._calculate_sample_weight(engagement)
                    sample_weights.append(weight)
                    
                except Keyframe.DoesNotExist:
                    continue
        
        if len(features) < self.min_training_samples:
            logger.info(f"Not enough training data for {query_type}: {len(features)} samples")
            return None, None, None
            
        return np.array(features), np.array(labels), np.array(sample_weights)
    
    def _calculate_sample_weight(self, engagement) -> float:
        """
        Calculate reliability weight for a training sample.
        More reliable engagement signals get higher weights.
        """
        weight = 1.0
        
        # Boost weight for strong engagement signals (realistic for video search)
        if engagement.video_played:
            weight += 1.0
        if engagement.video_completion_rate > 0.5:
            weight += 1.0
        if engagement.video_opened_fullscreen:
            weight += 1.5  # Very strong signal
        if engagement.was_used_as_filter:
            weight += 2.0  # Strongest signal - functional use
        if engagement.clip_copied:
            weight += 1.2  # Strong utility signal
        if engagement.video_scrubbed:
            weight += 0.8  # Active exploration
        if engagement.related_searches_performed > 0:
            weight += min(1.0, engagement.related_searches_performed * 0.3)  # Inspired further searches
            
        # Reduce weight for potentially noisy signals
        if engagement.was_clicked and engagement.view_duration < 1.0:
            weight *= 0.5  # Quick exit = less reliable
        if engagement.result_position > 10:
            weight *= 0.8  # Results far down the list = less attention
            
        # Boost weight for longer view times (more deliberate)
        if engagement.view_duration >= 10:
            weight += 0.5
        elif engagement.view_duration >= 30:
            weight += 1.0
            
        return max(0.1, weight)  # Minimum weight of 0.1
    
    def _extract_feature_similarities(self, query: str, kf_features: dict, query_type: str) -> List[float]:
        """
        Extract similarity scores between query and keyframe for each feature type.
        This would ideally use stored query embeddings, but for now compute on-the-fly.
        """
        from VideoSearch.utils.combined_embeddings import get_combined_vector_builder
        from utils.search import VideoSearchIndex  # Would need to import properly
        
        similarities = []
        
        # This is simplified - in practice we'd store query embeddings
        try:
            # Placeholder similarity scores (would compute actual similarities)
            clip_sim = np.random.random()  # Placeholder
            transcript_sim = np.random.random()  # Placeholder  
            dino_sim = np.random.random()  # Placeholder
            object_sim = np.random.random()  # Placeholder
            color_sim = np.random.random()  # Placeholder
            
            similarities = [clip_sim, transcript_sim, dino_sim, object_sim, color_sim]
            
        except Exception as e:
            # Fallback to neutral similarities
            similarities = [0.5] * 5
            
        return similarities
    
    def train_weights(self, query_type: str, retrain: bool = False):
        """
        Train optimal weights for a query type using engagement data.
        
        Uses weighted linear regression to learn which feature similarities predict
        high engagement. Feature coefficients become the new weights.
        """
        from VideoSearch.models import WeightingModel
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Check if we need to retrain
        try:
            model = WeightingModel.objects.get(query_type=query_type)
            if not retrain and model.training_samples >= self.min_training_samples:
                recent_update = timezone.now() - timedelta(days=7)
                if model.last_updated >= recent_update:
                    logger.info(f"Weights for {query_type} are recent, skipping training")
                    return model.get_weights_dict()
        except WeightingModel.DoesNotExist:
            model = WeightingModel.objects.create(query_type=query_type)
        
        # Prepare training data
        X, y, sample_weights = self.prepare_training_data(query_type)
        if X is None:
            logger.info(f"Insufficient data to train {query_type} weights")
            return model.get_weights_dict()
        
        # Train regression model
        try:
            # Normalize features
            if query_type not in self.scalers:
                self.scalers[query_type] = StandardScaler()
            
            X_scaled = self.scalers[query_type].fit_transform(X)
            
            # Train weighted regression model
            reg_model = LinearRegression()
            reg_model.fit(X_scaled, y, sample_weight=sample_weights)
            
            # Extract feature weights (coefficients indicate importance)
            coefficients = reg_model.coef_
            
            # Convert coefficients to positive weights with normalization
            feature_weights = np.abs(coefficients[:5])  # First 5 are feature similarities
            
            # Prevent division by zero and ensure reasonable scaling
            if np.sum(feature_weights) > 0:
                feature_weights = feature_weights / np.sum(feature_weights) * 5  # Normalize to reasonable scale
            else:
                # Fallback to default weights if all coefficients are zero
                feature_weights = np.array([1.0, 0.8, 0.6, 0.4, 0.3])
            
            # Update model weights
            new_weights = {
                'clip_emb': float(feature_weights[0]),
                'transcript_embedding': float(feature_weights[1]), 
                'dino_emb': float(feature_weights[2]),
                'object_vector': float(feature_weights[3]),
                'histogram': float(feature_weights[4]),
            }
            
            model.update_weights(new_weights)
            model.training_samples = len(X)
            
            # Calculate engagement-based performance metrics
            predictions = reg_model.predict(X_scaled)
            
            # Engagement metrics
            avg_engagement = float(np.average(y, weights=sample_weights))
            high_engagement_rate = float(np.mean(y > 0.5))
            model_r2 = float(r2_score(y, predictions, sample_weight=sample_weights))
            
            model.average_engagement_score = avg_engagement
            model.high_engagement_rate = high_engagement_rate
            
            # Legacy click rate for backwards compatibility
            binary_clicks = (y > 0.3).astype(float)  # Threshold for "clicked"
            model.click_through_rate = float(np.average(binary_clicks, weights=sample_weights))
            
            model.save()
            
            logger.info(f"Updated {query_type} weights: {new_weights}")
            logger.info(f"Training samples: {len(X)}, Avg engagement: {avg_engagement:.3f}, "
                       f"High engagement rate: {high_engagement_rate:.3f}, R²: {model_r2:.3f}")
            
            return new_weights
            
        except Exception as e:
            logger.error(f"Failed to train weights for {query_type}: {e}")
            return model.get_weights_dict()
    
    def get_smart_weights(self, query_type: str) -> Dict[str, float]:
        """
        Get the current smart weights for a query type.
        
        Returns learned weights if available, otherwise falls back to defaults.
        """
        from VideoSearch.models_analytics import WeightingModel
        
        try:
            model = WeightingModel.objects.get(query_type=query_type)
            
            # Check if we should trigger retraining
            if model.training_samples < self.min_training_samples:
                # Try to train if we have enough new data
                return self.train_weights(query_type)
            
            # Check for periodic retraining
            week_ago = timezone.now() - timedelta(days=7)
            if model.last_updated < week_ago:
                return self.train_weights(query_type)
                
            return model.get_weights_dict()
            
        except WeightingModel.DoesNotExist:
            # Train initial model
            return self.train_weights(query_type)
    
    def update_combined_builder_weights(self, query_type: str):
        """
        Update the combined vector builder with smart weights for a query type.
        """
        from VideoSearch.utils.combined_embeddings import get_combined_vector_builder
        
        builder = get_combined_vector_builder()
        smart_weights = self.get_smart_weights(query_type)
        
        # Update builder weights (temporarily)
        original_weights = builder.weights.copy()
        builder.weights.update(smart_weights)
        
        return original_weights  # Return for restoration later

# Global instance
_smart_weighting_system = None

def get_smart_weighting_system() -> SmartWeightingSystem:
    """Get shared smart weighting system instance."""
    global _smart_weighting_system
    if _smart_weighting_system is None:
        _smart_weighting_system = SmartWeightingSystem()
    return _smart_weighting_system