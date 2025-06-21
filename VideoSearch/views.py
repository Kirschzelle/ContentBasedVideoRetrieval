from django.shortcuts import render, get_object_or_404
from .models import Clip

# Create your views here.
def home_view(request):
    query = request.GET.get("q")
    clips = []
    video_ids = []
    
    if query:
        clips = Clip.objects.filter(video__file_path__icontains=query)
        seen_urls = []
        
        for clip in clips:
            if clip.video.media_url not in seen_urls:
                seen_urls.append(clip.video.media_url)
        video_ids = seen_urls

    context = {
        "query": query,
        "clips": clips,
        "start_time": 0,
        "video_ids": video_ids
    }

    return render(request, "home.html", context)

def detailed_view(request, start_frame): #TODO: Change to keyframe_id
    clip = Clip.objects.filter(start_frame=start_frame).first() #TODO: replace this with get_object_or_404(keyframe_id=keyframe_id)
    context = {
        "clip": clip,
    }
    return render(request, "detailed_view.html", context)