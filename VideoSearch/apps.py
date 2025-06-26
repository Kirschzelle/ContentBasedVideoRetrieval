from django.apps import AppConfig


class VideosearchConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'VideoSearch'

    def ready(self):
        import VideoSearch.signals
