"""
Integration layer for smart weighting system with the search functionality.

This module shows how to integrate the smart weighting system with the existing
search infrastructure to provide adaptive, learning-based search relevance.
"""

import logging
from typing import List, Dict, Optional
import time
import uuid
from django.utils import timezone

logger = logging.getLogger(__name__)

class SmartSearchIntegration:
    """
    Integration wrapper that combines the search system with smart weighting.
    
    This class wraps the existing VideoSearchIndex to add:
    1. User interaction tracking
    2. Smart weight adaptation
    3. Performance monitoring
    """
    
    def __init__(self, search_index):
        """Initialize with existing search index."""
        self.search_index = search_index
        self.active_sessions = {}  # Track ongoing search sessions
    
    def smart_search(self, query: str, returned_ids=None, filters=None, 
                    top_k=5, session_id: str = None, track_interaction: bool = True):
        """
        Perform search with smart weighting and interaction tracking.
        
        Args:
            query: Search query string
            returned_ids: Previously returned keyframe IDs to exclude
            filters: Filter parameters
            top_k: Number of results to return
            session_id: Optional session ID for tracking
            track_interaction: Whether to track this search for learning
            
        Returns:
            (results, interaction_id) where interaction_id can be used to track clicks
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Determine query type (could be enhanced with NLP)
        query_type = self._classify_query_type(query)
        
        # Apply smart weights if available
        smart_weights_applied = False
        original_weights = None
        
        if self.search_index.combined_index is not None:
            try:
                from VideoSearch.utils.smart_weighting import get_smart_weighting_system
                smart_system = get_smart_weighting_system()
                original_weights = smart_system.update_combined_builder_weights(query_type)
                smart_weights_applied = True
            except Exception as e:
                logger.warning(f"Failed to apply smart weights: {e}")
        
        # Perform the search
        search_start = time.time()
        results = self.search_index.search_incremental(
            query, returned_ids, filters, top_k
        )
        search_duration = time.time() - search_start
        
        # Restore original weights
        if smart_weights_applied and original_weights:
            try:
                from VideoSearch.utils.combined_embeddings import get_combined_vector_builder
                builder = get_combined_vector_builder()
                builder.weights = original_weights
            except Exception as e:
                logger.warning(f"Failed to restore original weights: {e}")
        
        # Track interaction if enabled
        interaction_id = None
        if track_interaction and results:
            try:
                from VideoSearch.utils.smart_weighting import get_smart_weighting_system
                smart_system = get_smart_weighting_system()
                interaction_id = smart_system.collect_search_interaction(
                    query=query,
                    query_type=query_type,
                    results=results,
                    filters=filters,
                    session_id=session_id
                )
                
                # Store session info for click tracking
                self.active_sessions[interaction_id] = {
                    'session_id': session_id,
                    'search_time': time.time(),
                    'query': query,
                    'results': [kf.id for kf in results]
                }
                
            except Exception as e:
                logger.warning(f"Failed to track search interaction: {e}")
        
        logger.info(f"Smart search completed: query='{query}', type={query_type}, "
                   f"results={len(results)}, duration={search_duration:.3f}s, "
                   f"smart_weights={smart_weights_applied}")
        
        return results, interaction_id
    
    def track_result_engagement(self, interaction_id: int, keyframe_id: int,
                              engagement_data: dict):
        """
        Track detailed user engagement with a search result.
        
        Args:
            interaction_id: ID from smart_search() call
            keyframe_id: ID of the keyframe that was engaged with
            engagement_data: Dict with engagement metrics (see SmartWeightingSystem.update_result_engagement)
        """
        if interaction_id not in self.active_sessions:
            logger.warning(f"Unknown interaction ID: {interaction_id}")
            return
            
        session_info = self.active_sessions[interaction_id]
        
        # Add time-based context if not provided
        if 'time_to_click' not in engagement_data and engagement_data.get('was_clicked'):
            engagement_data['time_to_click'] = time.time() - session_info['search_time']
        
        try:
            from VideoSearch.utils.smart_weighting import get_smart_weighting_system
            smart_system = get_smart_weighting_system()
            smart_system.update_result_engagement(interaction_id, keyframe_id, engagement_data)
            
            logger.info(f"Updated engagement: keyframe={keyframe_id}, "
                       f"clicked={engagement_data.get('was_clicked', False)}, "
                       f"view_time={engagement_data.get('view_duration', 0):.1f}s")
            
        except Exception as e:
            logger.error(f"Failed to update engagement: {e}")
    
    def track_result_click(self, interaction_id: int, clicked_keyframe_id: int,
                          result_position: int, view_duration: float = None,
                          next_action: str = None):
        """
        Legacy method - track that user clicked on a search result.
        
        For richer tracking, use track_result_engagement() instead.
        """
        engagement_data = {
            'was_clicked': True,
            'view_duration': view_duration or 0.0,
            'next_action': next_action
        }
        self.track_result_engagement(interaction_id, clicked_keyframe_id, engagement_data)
    
    def track_video_engagement(self, interaction_id: int, keyframe_id: int,
                             watch_duration: float, completion_rate: float):
        """
        Track video watching engagement.
        
        Args:
            interaction_id: Search interaction ID
            keyframe_id: Keyframe that contains the video
            watch_duration: Seconds of video watched
            completion_rate: Fraction of video watched (0.0-1.0)
        """
        engagement_data = {
            'video_played': True,
            'video_watch_duration': watch_duration,
            'video_completion_rate': completion_rate
        }
        self.track_result_engagement(interaction_id, keyframe_id, engagement_data)
    
    def track_result_action(self, interaction_id: int, keyframe_id: int,
                           action: str, **kwargs):
        """
        Track specific actions taken on search results.
        
        Args:
            interaction_id: Search interaction ID
            keyframe_id: Keyframe the action was taken on
            action: Type of action (download, share, bookmark, filter)
            **kwargs: Additional engagement data
        """
        engagement_data = kwargs.copy()
        
        if action == 'copy_clip':
            engagement_data['clip_copied'] = True
        elif action == 'fullscreen':
            engagement_data['video_opened_fullscreen'] = True
        elif action == 'filter':
            engagement_data['was_used_as_filter'] = True
        elif action == 'scrub':
            engagement_data['video_scrubbed'] = True
        elif action == 'related_search':
            engagement_data['related_searches_performed'] = engagement_data.get('related_searches_performed', 0) + 1
        
        engagement_data['next_action'] = action
        self.track_result_engagement(interaction_id, keyframe_id, engagement_data)
    
    def trigger_weight_training(self, query_type: str = None, force: bool = False):
        """
        Manually trigger weight training for specific or all query types.
        
        Args:
            query_type: Specific query type to train, or None for all
            force: Whether to force retraining even if recent
        """
        try:
            from VideoSearch.utils.smart_weighting import get_smart_weighting_system
            from VideoSearch.models import WeightingModel
            
            smart_system = get_smart_weighting_system()
            
            if query_type:
                query_types = [query_type]
            else:
                # Train all known query types
                query_types = ['balanced', 'visual', 'text']
                # Add any custom query types from database
                existing_types = WeightingModel.objects.values_list('query_type', flat=True)
                query_types.extend(existing_types)
                query_types = list(set(query_types))  # Remove duplicates
            
            results = {}
            for qtype in query_types:
                logger.info(f"Training weights for query type: {qtype}")
                weights = smart_system.train_weights(qtype, retrain=force)
                results[qtype] = weights
                
            return results
            
        except Exception as e:
            logger.error(f"Failed to train weights: {e}")
            return {}
    
    def get_performance_metrics(self):
        """Get performance metrics for the smart weighting system."""
        try:
            from VideoSearch.models import WeightingModel, SearchInteraction, ResultClick
            
            metrics = {}
            
            # Overall stats
            total_searches = SearchInteraction.objects.count()
            total_clicks = ResultClick.objects.count()
            overall_ctr = (total_clicks / total_searches) if total_searches > 0 else 0
            
            metrics['overall'] = {
                'total_searches': total_searches,
                'total_clicks': total_clicks,
                'click_through_rate': overall_ctr
            }
            
            # Per query type stats
            for model in WeightingModel.objects.all():
                metrics[model.query_type] = {
                    'training_samples': model.training_samples,
                    'click_through_rate': model.click_through_rate,
                    'last_updated': model.last_updated.isoformat() if model.last_updated else None,
                    'weights': model.get_weights_dict()
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    def _classify_query_type(self, query: str) -> str:
        """
        Classify query type based on content.
        
        This is a simple heuristic-based approach. Could be enhanced with
        trained NLP models for better classification.
        """
        query_lower = query.lower()
        
        # Text-focused indicators
        text_keywords = ['said', 'says', 'talking', 'speak', 'voice', 'audio', 
                        'transcript', 'word', 'mention', 'discuss']
        
        # Visual-focused indicators  
        visual_keywords = ['color', 'bright', 'dark', 'scene', 'show', 'see',
                          'appear', 'visual', 'look', 'image', 'picture']
        
        text_score = sum(1 for keyword in text_keywords if keyword in query_lower)
        visual_score = sum(1 for keyword in visual_keywords if keyword in query_lower)
        
        if text_score > visual_score and text_score >= 1:
            return 'text'
        elif visual_score > text_score and visual_score >= 1:
            return 'visual'
        else:
            return 'balanced'
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old session data to prevent memory leaks."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        old_sessions = [
            session_id for session_id, info in self.active_sessions.items()
            if info['search_time'] < cutoff_time
        ]
        
        for session_id in old_sessions:
            del self.active_sessions[session_id]
            
        if old_sessions:
            logger.info(f"Cleaned up {len(old_sessions)} old sessions")

# Usage example integration
def create_smart_search_wrapper(search_index):
    """
    Create a smart search wrapper around an existing VideoSearchIndex.
    
    Example usage:
        # In your search view or API
        smart_search = create_smart_search_wrapper(your_search_index)
        results, interaction_id = smart_search.smart_search("dogs playing", top_k=10)
        
        # Later, when user clicks on result
        smart_search.track_result_click(interaction_id, keyframe_id, position)
        
        # Periodically retrain weights
        smart_search.trigger_weight_training()
    """
    return SmartSearchIntegration(search_index)