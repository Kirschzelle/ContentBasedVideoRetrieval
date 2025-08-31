"""
URL patterns for external frame upload functionality.
"""

from django.urls import path
from VideoSearch import views

app_name = 'external_frame'

urlpatterns = [
    # Frame upload endpoints
    path('upload/', views.upload_frame_for_search, name='upload_frame'),
    path('features/', views.get_external_frame_features, name='get_features'),
    path('clear/', views.clear_external_frame_features, name='clear_features'),
    
    # Upload page for testing
    path('', views.external_frame_upload_page, name='upload_page'),
]