from django.shortcuts import render, get_object_or_404
from django.conf import settings
from django.http import JsonResponse
from pathlib import Path
from .models import Keyframe
from collections import defaultdict
import sys
import os
from django.utils.http import urlencode

_searcher_instance = None

def get_searcher():
    global _searcher_instance

    if _searcher_instance is None:
        from utils.search import Searcher  # Lazy import
        _searcher_instance = Searcher()
    return _searcher_instance

# Create your views here.
def home_view(request):
    query = request.GET.get("q", "")
    context = {
        "query": query,
        "clips": [],      # optional for JS-based rendering
        "video_ids": []   # can preload later if needed
    }
    return render(request, "home.html", context)


def api_search_view(request):
    query = request.GET.get("q")
    returned = request.GET.getlist("returned[]")
    returned_ids = set(map(int, returned)) if returned else set()

    if not query:
        return JsonResponse({"error": "No query provided."}, status=400)

    # Check for special uploaded frames query
    if query == "uploaded_frame:latest":
        from VideoSearch.models import UploadedFrame
        uploaded_frames = UploadedFrame.objects.all()[:20]  # Show last 20 uploaded frames
        
        keyframe_data = []
        for frame in uploaded_frames:
            keyframe_data.append({
                "keyframe_id": f"uploaded_{frame.id}",
                "thumbnail": f"/media/uploaded_frames/{frame.filename}",  # Fake URL for display
                "is_uploaded_frame": True,
                "upload_timestamp": frame.upload_timestamp.isoformat()
            })
        
        return JsonResponse({"results": keyframe_data})

    filters = defaultdict(list)
    filters_raw = request.GET.getlist("filters[]")

    for pair in filters_raw:
        try:
            kf_id_str, category = pair.split(":")
            kf_id = int(kf_id_str)
            filters[kf_id].append(category)
        except ValueError:
            continue

    results = get_searcher().search_incremental(query, returned_ids=returned_ids, filters=filters, top_k=1000)
    if not results:
        return JsonResponse({"done": True})

    media_root = Path(settings.MEDIA_ROOT).resolve()
    keyframe_data = []

    for kf in results:
        image_path = kf.get_image_path().resolve()

        try:
            relative_path = image_path.relative_to(media_root)
        except ValueError:
            continue  # skip invalid

        image_url = settings.MEDIA_URL.rstrip("/") + "/" + str(relative_path).replace("\\", "/")

        keyframe_data.append({
            "keyframe_id": kf.id,
            "thumbnail": image_url
        })

    return JsonResponse({"results": keyframe_data})

def detailed_view(request, keyframe_id):
    query = request.GET.get('q', '')
    keyframe = get_object_or_404(Keyframe, id=keyframe_id)
    image_path = keyframe.get_image_path().resolve()
    media_root = Path(settings.MEDIA_ROOT).resolve()

    try:
        relative_path = image_path.relative_to(media_root)
    except ValueError:
        return JsonResponse({"error": "Image path is not within MEDIA_ROOT"}, status=500)

    image_url = settings.MEDIA_URL.rstrip("/") + "/" + str(relative_path).replace("\\", "/")

    # Add this to preserve query + filters
    query_string = urlencode(request.GET, doseq=True)

    context = {
        "keyframe": keyframe,
        "keyframe_img": image_url,
        "query": query,
        "query_string": query_string,  # ← added
    }
    return render(request, "detailed_view.html", context)

# DaVinci Resolve Integration Views
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import logging

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
        from VideoSearch.utils.davinci_integration import send_clip_to_davinci_preview
        result = send_clip_to_davinci_preview(keyframe)
        
        # Track engagement if successful (this is a strong signal!)
        if result['success']:
            try:
                logger.info(f"User sent keyframe {keyframe_id} to DaVinci - strong engagement signal")
            except Exception as e:
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
    Check if DaVinci Resolve is available and ready via bridge server.
    """
    from VideoSearch.utils.davinci_integration import check_davinci_status as check_status
    result = check_status()
    return JsonResponse(result)

# External Frame Upload Views

@csrf_exempt
@require_POST
def upload_frame_for_search(request):
    """
    Upload an external frame (from DaVinci export, etc.) and extract features for search.
    
    POST parameters:
    - frame_image: Uploaded image file
    """
    try:
        if 'frame_image' not in request.FILES:
            return JsonResponse({
                'success': False,
                'error': 'No frame image provided'
            })
        
        uploaded_file = request.FILES['frame_image']
        
        # Validate file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff']
        if uploaded_file.content_type not in allowed_types:
            return JsonResponse({
                'success': False,
                'error': f'Unsupported file type: {uploaded_file.content_type}. Use JPG, PNG, or TIFF.'
            })
        
        # Validate file size (max 50MB)
        if uploaded_file.size > 50 * 1024 * 1024:
            return JsonResponse({
                'success': False,
                'error': 'File too large. Maximum size is 50MB.'
            })
        
        # Process the frame
        logger.info(f"Processing uploaded frame: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        from VideoSearch.utils.external_frame_processing import process_external_frame_for_search
        result = process_external_frame_for_search(uploaded_file)
        
        if result['success']:
            # Store frame in database
            from VideoSearch.models import UploadedFrame
            
            features = result['features']
            image_info = result['image_info']
            
            # Save uploaded frame to database
            uploaded_frame = UploadedFrame(
                filename=uploaded_file.name,
                file_size=uploaded_file.size,
                content_type=uploaded_file.content_type,
                width=image_info['width'],
                height=image_info['height'],
                session_key=request.session.session_key,
                embedding_clip=UploadedFrame.compress_array(features['clip_emb']),
                embedding_dino=UploadedFrame.compress_array(features['dino_emb']) if features.get('dino_emb') is not None else None,
                histogram_hsv=UploadedFrame.compress_array(features['histogram']) if features.get('histogram') is not None else None,
                dominant_colors=UploadedFrame.compress_array(features['palette']) if features.get('palette') is not None else None,
                colorfulness=features.get('colorfulness'),
                object_vector=UploadedFrame.compress_array(features['object_vector']) if features.get('object_vector') is not None else None,
            )
            uploaded_frame.save()
            
            return JsonResponse({
                'success': True,
                'message': f'Frame processed successfully: {uploaded_file.name}',
                'image_info': image_info,
                'features_available': list(features.keys()),
                'redirect_to': '/?q=uploaded_frame:latest'  # Auto-redirect to special query
            })
        else:
            return JsonResponse({
                'success': False,
                'error': result['error']
            })
            
    except Exception as e:
        logger.error(f"External frame upload error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': f'Server error: {str(e)}'
        })

def get_external_frame_features(request):
    """
    Get features from previously uploaded external frame.
    Used by search system to apply external frame as filter.
    """
    try:
        if 'external_frame_features' not in request.session:
            return JsonResponse({
                'success': False,
                'error': 'No external frame features available. Upload a frame first.'
            })
        
        frame_data = request.session['external_frame_features']
        
        return JsonResponse({
            'success': True,
            'features': frame_data['features'],
            'image_info': frame_data['image_info']
        })
        
    except Exception as e:
        logger.error(f"Get external frame features error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

def clear_external_frame_features(request):
    """Clear external frame features from session."""
    try:
        if 'external_frame_features' in request.session:
            del request.session['external_frame_features']
        
        return JsonResponse({
            'success': True,
            'message': 'External frame features cleared'
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

def external_frame_upload_page(request):
    """Simple upload page for testing external frame functionality."""
    return render(request, 'external_frame_upload.html')