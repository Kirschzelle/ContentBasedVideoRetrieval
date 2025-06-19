from django.shortcuts import render
from .models import Clip

# Create your views here.
def home_view(request):
    query = request.GET.get("q")
    clips = []
    if query:
        clips = Clip.objects.filter(video__file_path__icontains=query)
    context = {
        "query": query,
        "clips": clips,
        "start_time": 0
    }
    return render(request, "home.html", context)

def detailed_view(request):
    context = {
        "test": "test",
    }
    return render(request, "home.html", context)