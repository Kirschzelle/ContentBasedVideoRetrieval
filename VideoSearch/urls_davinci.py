"""
URL patterns for DaVinci Resolve integration.
"""

from django.urls import path
from VideoSearch.views import davinci_views

app_name = 'davinci'

urlpatterns = [
    path('send/', davinci_views.send_to_davinci, name='send_to_davinci'),
    path('status/', davinci_views.check_davinci_status, name='check_status'),
]