"""
URL patterns for DaVinci Resolve integration.
"""

from django.urls import path
from VideoSearch.views import send_to_davinci, check_davinci_status, get_current_frame_from_davinci

app_name = 'davinci'

urlpatterns = [
    path('send/', send_to_davinci, name='send_to_davinci'),
    path('status/', check_davinci_status, name='check_status'),
    path('get_current_frame/', get_current_frame_from_davinci, name='get_current_frame'),
]