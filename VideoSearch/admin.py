from django.contrib import admin
from .models import Keyframe, Video, Clip, ClipPredictionCache

# Register your models here.
admin.site.register(Video)
admin.site.register(Clip)
admin.site.register(ClipPredictionCache)
admin.site.register(Keyframe)