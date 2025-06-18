from django.shortcuts import render

# Create your views here.
def home_view(request):
    context = {
        "test": "test",
    }
    return render(request, "home.html", context)