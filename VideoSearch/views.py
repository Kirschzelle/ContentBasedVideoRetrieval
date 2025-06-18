from django.shortcuts import render

# Create your views here.
def home_view(request):
    context = {
        "start_time": 120,
    }
    return render(request, "home.html", context)

def detailed_view(request):
    context = {
        "test": "test",
    }
    return render(request, "home.html", context)