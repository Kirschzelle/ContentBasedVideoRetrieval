from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import Counter
from typing import List, Set
from VideoSearch.models import Keyframe

# YOLOv8 COCO class names (80 classes)
YOLO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

class ObjectDetector:
    def __init__(self, model_name="yolov8x.pt", command=None, conf_threshold=0.05):
        self.command = command
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

    def _log(self, msg, level="info"):
        if self.command:
            self.command.stdout.write(getattr(self.command, f"style_{level}")(msg))
        else:
            print(msg)

    def extract_objects(self, image: Image.Image) -> dict:
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        label_conf = {}

        for r in results:
            for cls_id, score in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                cls_name = YOLO_CLASSES[int(cls_id)]
                if cls_name not in label_conf or score > label_conf[cls_name]:
                    label_conf[cls_name] = float(score)

        return {"objects": label_conf.copy()}

    def extract_objects_batch(self, images: List[Image.Image]) -> List[dict]:
        batch_results = self.model.predict(images, conf=self.conf_threshold, verbose=False)
        results = []

        for r in batch_results:
            label_conf = {}
            for cls_id, score in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                cls_name = YOLO_CLASSES[int(cls_id)]
                if cls_name not in label_conf or score > label_conf[cls_name]:
                    label_conf[cls_name] = float(score)
            results.append(label_conf)

        return [{"objects": obj} for obj in results]

# ---------- Comparison Logic ----------

def soft_object_distance(a: dict, b: dict) -> float:
    """
    Soft Jaccard distance using confidence scores.
    """
    if not "objects" in a or not "objects" in b:
        return 1.0

    intersection = 0.0
    union = 0.0

    all_keys = set(a["objects"].keys()) | set(b["objects"].keys())
    for k in all_keys:
        val_a = a["objects"].get(k, 0.0)
        val_b = b["objects"].get(k, 0.0)
        intersection += min(val_a, val_b)
        union += max(val_a, val_b)

    return 1.0 - (intersection / union) if union > 0 else 1.0

def distance_to_existing_keyframes(clip, query_labels: dict):
    keyframes = Keyframe.objects.filter(clip=clip)
    if not keyframes.exists():
        return 1.0, 1.0

    distances = []
    for kf in keyframes:
        labels = kf.load_object_labels()
        if labels:
            distances.append(soft_object_distance(query_labels, {"objects": labels}))
    return min(distances), max(distances)
