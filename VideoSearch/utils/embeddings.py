import torch
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel
import timm
from PIL import Image
from VideoSearch.utils.hardware import EmbeddingModelSelector
from VideoSearch.models import Keyframe
import numpy as np
from scipy.spatial.distance import cosine

class ImageEmbedder:
    def __init__(self, device=None, command=None):
        self.command = command
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        clip_model_name, dino_model_name, mode = EmbeddingModelSelector.select(command=self.command)
        self.mode = mode

        self._log(f"[Embedding] Loading CLIP model: {clip_model_name}", "info")
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        self.dino_model = None
        if mode != "clip-only":
            self._log(f"[Embedding] Loading DINO model: {dino_model_name}", "info")
            self.dino_model = timm.create_model(dino_model_name, pretrained=True).to(self.device)
            self.dino_model.eval()

        self._log(f"[Embedding] Initialized mode: {mode} | Device: {self.device}", "success")

        self.dino_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def _log(self, msg, level="info"):
        if self.command:
            self.command.stdout.write(getattr(self.command, f"style_{level}")(msg))
        else:
            print(msg)

    def get_clip_embedding(self, image: Image.Image):
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.clip_model.get_image_features(**inputs)
        emb = emb[0]
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.detach().cpu().numpy()

    def get_dino_embedding(self, image: Image.Image):
        if self.dino_model is None:
            return None
        img_tensor = self.dino_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.dino_model(img_tensor)
        emb = emb[0]
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.detach().cpu().numpy()

    def get_combined_embedding(self, image: Image.Image) -> dict:
        features = {
            "clip_emb": self.get_clip_embedding(image)
        }

        if self.mode != "clip-only":
            dino_emb = self.get_dino_embedding(image)
            features["dino_emb"] = dino_emb
        else:
            features["dino_emb"] = None

        return features

    def calculate_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Computes cosine distance (1 - cosine similarity) between two normalized embeddings.
        """
        return float(cosine(emb1, emb2))

    def calculate_combined_distance(self, a: dict, b: dict) -> float:
        distances = []

        if "clip_emb" in a and "clip_emb" in b:
            distances.append(self.calculate_distance(a["clip_emb"], b["clip_emb"]))

        if "dino_emb" in a and "dino_emb" in b and a["dino_emb"] is not None and b["dino_emb"] is not None:
            distances.append(self.calculate_distance(a["dino_emb"], b["dino_emb"]))

        if not distances:
            raise ValueError("No valid embeddings provided for distance calculation.")

        return float(np.mean(distances))

    def get_combined_distance_to_set(self, query: dict, feature_set: list[dict]):
        """
        Given one embedding dict and a list of others, compute min/max combined distances.
        """
        distances = [self.calculate_combined_distance(query, f) for f in feature_set]
        return min(distances), max(distances)

    def get_all_combined_distances(self, queries: list[dict], references: list[dict]) -> list[list[float]]:
        return [
            [self.calculate_combined_distance(q, r) for r in references]
            for q in queries
        ]

    def get_distance_to_existing_keyframes(self, clip, query_features: dict):
        """
        Computes min/max distance from the given embedding dict to keyframes in the clip.
        Returns (min_distance, max_distance). If no keyframes exist, returns (1.0, 1.0).
        """
        keyframes = Keyframe.objects.filter(clip=clip)
        if not keyframes.exists():
            return 1.0, 1.0

        distances = []
        for kf in keyframes:
            features_kf = {
                "clip_emb": kf.load_embedding_clip(),
                "dino_emb": kf.load_embedding_dino()
            }
            dist = self.calculate_combined_distance(query_features, features_kf)
            distances.append(dist)

        return min(distances), max(distances)