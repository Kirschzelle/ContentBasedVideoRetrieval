from django.contrib import admin
from .models import MediaFolderSetting, Keyframe, Video, Clip, ClipPredictionCache

@admin.register(MediaFolderSetting)
class MediaFolderSettingAdmin(admin.ModelAdmin):
    list_display = ('name', 'path', 'recursive', 'is_active', 'created_at')
    list_filter = ('is_active', 'recursive', 'created_at')
    search_fields = ('name', 'path')
    readonly_fields = ('created_at', 'updated_at')

@admin.register(Video)  
class VideoAdmin(admin.ModelAdmin):
    list_display = ('id', 'file_name', 'resolution', 'frame_count', 'fps_display')
    list_filter = ('resolution',)
    search_fields = ('file_path',)
    readonly_fields = ('file_path', 'web_path', 'frame_count', 'fps_num', 'fps_den', 'resolution')
    
    def fps_display(self, obj):
        return f"{obj.fps():.2f}"
    fps_display.short_description = 'FPS'

# Register your models here.
admin.site.register(Clip)
admin.site.register(ClipPredictionCache)
admin.site.register(Keyframe)