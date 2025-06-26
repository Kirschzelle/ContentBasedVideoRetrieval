import os
from pathlib import Path
from django.db.models.signals import post_delete
from django.dispatch import receiver
from .models import Keyframe, Clip

@receiver(post_delete, sender=Keyframe)
def delete_keyframe_image(sender, instance: Keyframe, **kwargs):
    image_path = instance.get_image_path()
    if image_path.exists():
        image_path.unlink()

    dir_path = image_path.parent
    if dir_path.exists() and not any(dir_path.iterdir()):
        dir_path.rmdir()

@receiver(post_delete, sender=Clip)
def delete_clip_keyframe_folder(sender, instance: Clip, **kwargs):
    from VideoSearch.models import KEYFRAME_ROOT 
    folder = KEYFRAME_ROOT / str(instance.id)
    if folder.exists() and not any(folder.iterdir()):
        folder.rmdir()