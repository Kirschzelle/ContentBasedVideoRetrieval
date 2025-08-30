"""
Django views for DaVinci Resolve integration.
"""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.shortcuts import get_object_or_404
import logging

from VideoSearch.models import Keyframe
from VideoSearch.utils.davinci_integration import send_clip_to_davinci_preview

logger = logging.getLogger(__name__)

@csrf_exempt
@require_POST
def send_to_davinci(request):
    """
    Send a clip to DaVinci Resolve preview.
    
    POST parameters:
    - keyframe_id: ID of the keyframe to send
    """
    try:
        keyframe_id = request.POST.get('keyframe_id')
        
        if not keyframe_id:
            return JsonResponse({
                'success': False,
                'error': 'No keyframe ID provided'
            })
        
        # Get keyframe with related data
        keyframe = get_object_or_404(
            Keyframe.objects.select_related('clip__video'),
            id=keyframe_id
        )
        
        # Send to DaVinci
        result = send_clip_to_davinci_preview(keyframe)
        
        # Track engagement if successful (this is a strong signal!)
        if result['success']:
            try:
                # This would integrate with smart weighting system
                # For now, we'll just log it
                logger.info(f"User sent keyframe {keyframe_id} to DaVinci - strong engagement signal")
                
                # Could track engagement here:
                # track_engagement(keyframe_id, 'davinci_preview', very_high_score)
                
            except Exception as e:
                # Don't let tracking errors break the main functionality
                logger.warning(f"Failed to track DaVinci engagement: {e}")
        
        return JsonResponse(result)
        
    except Exception as e:
        logger.error(f"DaVinci view error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': f'Server error: {str(e)}'
        })

def check_davinci_status(request):
    """
    Check if DaVinci Resolve is available and ready.
    """
    try:
        import DaVinciResolveScript as dvr
        resolve = dvr.scriptapp("Resolve")
        
        if resolve:
            project = resolve.GetProjectManager().GetCurrentProject()
            return JsonResponse({
                'available': True,
                'project_open': project is not None,
                'project_name': project.GetName() if project else None
            })
        else:
            return JsonResponse({
                'available': False,
                'error': 'DaVinci Resolve not running'
            })
            
    except ImportError:
        return JsonResponse({
            'available': False,
            'error': 'DaVinci Resolve API not installed'
        })
    except Exception as e:
        return JsonResponse({
            'available': False,
            'error': str(e)
        })