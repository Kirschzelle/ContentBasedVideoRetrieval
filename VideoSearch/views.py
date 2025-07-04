from django.shortcuts import render, get_object_or_404
from django.conf import settings
from django.http import JsonResponse
from pathlib import Path
from .models import Keyframe
from utils.search import Searcher
from collections import defaultdict
import sys
import os
from django.utils.http import urlencode

_searcher_instance = None

def get_searcher():
    global _searcher_instance

    if _searcher_instance is None and (
            "runserver" in sys.argv or
            "runserver_plus" in sys.argv
        ) and os.environ.get("RUN_MAIN") == "true":
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