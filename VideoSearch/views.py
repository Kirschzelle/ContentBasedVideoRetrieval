from django.shortcuts import render
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

def detailed_view(request):
    context = {
        "test": "test",
    }
    return render(request, "home.html", context)