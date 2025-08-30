from django.db import models
from django.contrib.auth.models import User
from .models import Keyframe
import json

class SearchInteraction(models.Model):
    """Track user search interactions for smart weighting system."""
    
    # Search context
    query = models.TextField()
    query_type = models.CharField(
        max_length=20, 
        choices=[('balanced', 'Balanced'), ('visual', 'Visual'), ('text', 'Text')],
        default='balanced'
    )
    search_timestamp = models.DateTimeField(auto_now_add=True)
    
    # User context (optional for anonymous usage)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_id = models.CharField(max_length=64, null=True, blank=True)
    
    # Search parameters 
    had_filters = models.BooleanField(default=False)
    filter_types = models.TextField(null=True, blank=True)  # JSON list of filter types used
    
    # Results and interaction
    total_results_returned = models.IntegerField()
    results_keyframe_ids = models.TextField()  # JSON list of returned keyframe IDs in order
    
    class Meta:
        indexes = [
            models.Index(fields=['search_timestamp']),
            models.Index(fields=['query_type']),
            models.Index(fields=['had_filters']),
        ]

class ResultClick(models.Model):
    """Track which search results users actually click on."""
    
    search_interaction = models.ForeignKey(SearchInteraction, on_delete=models.CASCADE)
    clicked_keyframe = models.ForeignKey(Keyframe, on_delete=models.CASCADE)
    
    # Click context
    result_position = models.IntegerField()  # 0-based position in search results
    click_timestamp = models.DateTimeField(auto_now_add=True)
    time_to_click = models.FloatField()  # Seconds from search to click
    
    # Interaction details
    view_duration = models.FloatField(null=True, blank=True)  # How long user viewed result
    next_action = models.CharField(
        max_length=20,
        choices=[
            ('new_search', 'New Search'),
            ('refine', 'Refine Search'), 
            ('filter', 'Apply Filter'),
            ('download', 'Download'),
            ('exit', 'Exit')
        ],
        null=True, blank=True
    )
    
    class Meta:
        unique_together = ('search_interaction', 'clicked_keyframe')
        indexes = [
            models.Index(fields=['result_position']),
            models.Index(fields=['click_timestamp']),
        ]

class WeightingModel(models.Model):
    """Store learned weighting parameters for different query types."""
    
    query_type = models.CharField(max_length=20, unique=True)
    
    # Core feature weights (learned from user interactions)
    clip_weight = models.FloatField(default=1.0)
    transcript_weight = models.FloatField(default=0.8) 
    dino_weight = models.FloatField(default=0.6)
    object_weight = models.FloatField(default=0.4)
    histogram_weight = models.FloatField(default=0.3)
    
    # Metadata
    training_samples = models.IntegerField(default=0)
    last_updated = models.DateTimeField(auto_now=True)
    model_version = models.CharField(max_length=20, default='1.0')
    
    # Performance metrics
    click_through_rate = models.FloatField(null=True, blank=True)
    average_result_position = models.FloatField(null=True, blank=True)  # Lower is better
    
    def get_weights_dict(self):
        """Return weights as dictionary compatible with CombinedVectorBuilder."""
        return {
            'clip_emb': self.clip_weight,
            'transcript_embedding': self.transcript_weight,
            'dino_emb': self.dino_weight, 
            'object_vector': self.object_weight,
            'histogram': self.histogram_weight,
        }
    
    def update_weights(self, new_weights: dict):
        """Update weights from training results."""
        self.clip_weight = new_weights.get('clip_emb', self.clip_weight)
        self.transcript_weight = new_weights.get('transcript_embedding', self.transcript_weight)
        self.dino_weight = new_weights.get('dino_emb', self.dino_weight)
        self.object_weight = new_weights.get('object_vector', self.object_weight)
        self.histogram_weight = new_weights.get('histogram', self.histogram_weight)
        self.save()
    
    class Meta:
        ordering = ['-last_updated']
    
    def __str__(self):
        return f"WeightingModel({self.query_type}) - {self.training_samples} samples"