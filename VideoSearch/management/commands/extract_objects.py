from django.core.management.base import BaseCommand
from VideoSearch.models import Keyframe
from VideoSearch.utils.objects import ObjectDetector
import json

class Command(BaseCommand):
    help = "Extract YOLO object features for keyframes missing them."

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=1,
            help='Number of keyframes to process in a batch (default: 1)'
        )

    def handle(self, *args, **options):
        batch_size = options['batch_size']
        detector = ObjectDetector(command=self)

        keyframes = list(Keyframe.objects.filter(object_labels__isnull=True))
        total = len(keyframes)

        if total == 0:
            self.stdout.write(self.style.SUCCESS("All keyframes already have object labels."))
            return

        self.stdout.write(self.style.WARNING(f"Found {total} keyframes missing object labels..."))
        self.stdout.write(f"Using batch size: {batch_size}")

        for i in range(0, total, batch_size):
            batch_kfs = keyframes[i:i + batch_size]
            images = []
            valid_kfs = []

            for kf in batch_kfs:
                img = kf.load_image()
                if img is not None:
                    images.append(img)
                    valid_kfs.append(kf)
                else:
                    self.stdout.write(self.style.ERROR(f"Skipping Keyframe {kf.id}: image missing"))

            if not images:
                continue

            features_batch = detector.extract_objects_batch(images)

            for kf, feat in zip(valid_kfs, features_batch):
                obj_data = feat.get("objects", {})
                if obj_data:
                    kf.object_labels = json.dumps(obj_data)
                    kf.save(update_fields=["object_labels"])
                    self.stdout.write(f"Keyframe {kf.id} processed.")
                else:
                    self.stdout.write(self.style.WARNING(f"No objects detected in Keyframe {kf.id}"))

        self.stdout.write(self.style.SUCCESS("Object extraction complete."))