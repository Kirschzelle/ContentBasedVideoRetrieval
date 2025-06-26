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

def label_dict_to_vector(label_conf: dict) -> np.ndarray:
    """
    Converts a YOLO label-confidence dict into an 80-dim numpy array.
    """
    vec = np.zeros(80, dtype=np.float32)
    for i, cls_name in enumerate(YOLO_CLASSES):
        vec[i] = label_conf.get(cls_name, 0.0)
    return vec

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

    def extract_vector(self, image: Image.Image) -> np.ndarray:
        """
        Returns an 80-dim object vector for a single image.
        """
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        label_conf = {}

        for r in results:
            for cls_id, score in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                cls_name = YOLO_CLASSES[int(cls_id)]
                if cls_name not in label_conf or score > label_conf[cls_name]:
                    label_conf[cls_name] = float(score)

        return label_dict_to_vector(label_conf)

    def extract_vector_batch(self, images: List[Image.Image]) -> List[np.ndarray]:
        """
        Returns a list of 80-dim vectors for a batch of images.
        """
        batch_results = self.model.predict(images, conf=self.conf_threshold, verbose=False)
        vectors = []

        for r in batch_results:
            label_conf = {}
            for cls_id, score in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                cls_name = YOLO_CLASSES[int(cls_id)]
                if cls_name not in label_conf or score > label_conf[cls_name]:
                    label_conf[cls_name] = float(score)
            vectors.append(label_dict_to_vector(label_conf))

        return vectors

# ---------- Comparison Logic ----------

def soft_object_distance(a: dict, b: dict) -> float:
    """
    Computes cosine distance between 'object_vec' vectors inside two feature dicts.
    Returns 1.0 if either vector is missing or zero.
    """
    vec_a = a.get("object_vec")
    vec_b = b.get("object_vec")

    if vec_a is None or vec_b is None:
        return 1.0

    return object_vector_distance(vec_a, vec_b)

def object_vector_distance(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 1.0
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return 1.0 - np.dot(a_norm, b_norm)

def distance_to_existing_keyframes(clip, query_vector: np.ndarray) -> tuple[float, float]:
    """
    Compares a query object vector to all keyframes in a clip.
    Returns (min_distance, max_distance), using cosine distance.
    """
    keyframes = Keyframe.objects.filter(clip=clip)
    if not keyframes.exists() or query_vector is None:
        return 1.0, 1.0

    distances = []
    for kf in keyframes:
        kf_vec = kf.load_object_vector()
        if kf_vec is not None:
            dist = object_vector_distance(query_vector, kf_vec)
            distances.append(dist)

    if not distances:
        return 1.0, 1.0

    return min(distances), max(distances)