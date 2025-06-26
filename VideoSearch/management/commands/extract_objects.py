from django.core.management.base import BaseCommand
from VideoSearch.models import Keyframe
from VideoSearch.utils.objects import ObjectDetector
import numpy as np

class Command(BaseCommand):
    help = "Extract YOLO object vectors for keyframes missing them."

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

        keyframes = list(Keyframe.objects.filter(object_vector__isnull=True))
        total = len(keyframes)

        if total == 0:
            self.stdout.write(self.style.SUCCESS("All keyframes already have object vectors."))
            return

        self.stdout.write(self.style.WARNING(f"Found {total} keyframes missing object vectors..."))
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

            # Use the new method returning 80-d vectors
            vectors_batch = detector.extract_vector_batch(images)

            for kf, vec in zip(valid_kfs, vectors_batch):
                if vec is not None and np.linalg.norm(vec) > 0:
                    kf.object_vector = Keyframe.compress_array(vec)
                    kf.save(update_fields=["object_vector"])
                    self.stdout.write(f"Keyframe {kf.id} processed.")
                else:
                    self.stdout.write(self.style.WARNING(f"No valid vector for Keyframe {kf.id}"))

        self.stdout.write(self.style.SUCCESS("Object vector extraction complete."))
