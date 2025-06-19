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

    def get_combined_embedding(self, image: Image.Image):
        clip_emb = self.get_clip_embedding(image)

        if self.mode == "clip-only":
            return clip_emb, None

        dino_emb = self.get_dino_embedding(image)
        return clip_emb, dino_emb

    def calculate_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Computes cosine distance (1 - cosine similarity) between two normalized embeddings.
        """
        return float(cosine(emb1, emb2))

    def calculate_combined_distance(
        self,
        clip1: np.ndarray, dino1: np.ndarray,
        clip2: np.ndarray, dino2: np.ndarray
    ) -> float:
        """
        Computes the average distance across available embeddings (CLIP, DINO).
        """
        distances = []
        if clip1 is not None and clip2 is not None:
            distances.append(self.calculate_distance(clip1, clip2))
        if dino1 is not None and dino2 is not None and self.mode != "clip-only":
            distances.append(self.calculate_distance(dino1, dino2))

        if not distances:
            raise ValueError("No valid embeddings provided for distance calculation.")

        return float(np.mean(distances))

    def get_combined_distance_to_set(self, clip_emb, dino_emb, set_clip_embs, set_dino_embs):
        """
        Computes the min/max combined distance from a given embedding to a set of embeddings.
        Skips DINO if not available.

        Returns: (min_distance, max_distance)
        """
        distances = []

        for i in range(len(set_clip_embs)):
            clip_dist = cosine(clip_emb, set_clip_embs[i])
            if dino_emb is not None and set_dino_embs[i] is not None:
                dino_dist = cosine(dino_emb, set_dino_embs[i])
                combined = np.mean([clip_dist, dino_dist])
            else:
                combined = clip_dist
            distances.append(combined)

        return min(distances), max(distances)

    def get_all_combined_distances(self, clip_embs, dino_embs, ref_clip_embs, ref_dino_embs):
        """
        Computes the combined distance from each embedding in the first list
        to all embeddings in the reference list. Returns a 2D list of distances.

        Returns: List[List[float]]
        """
        results = []
        for i in range(len(clip_embs)):
            row = []
            for j in range(len(ref_clip_embs)):
                clip_dist = cosine(clip_embs[i], ref_clip_embs[j])
                if dino_embs[i] is not None and ref_dino_embs[j] is not None:
                    dino_dist = cosine(dino_embs[i], ref_dino_embs[j])
                    combined = np.mean([clip_dist, dino_dist])
                else:
                    combined = clip_dist
                row.append(combined)
            results.append(row)
        return results

    def get_distance_to_existing_keyframes(self, clip, clip_emb: np.ndarray, dino_emb: np.ndarray):
        """
        Computes min/max distance from the given embedding to existing keyframes in the given clip.
        Returns (min_distance, max_distance). If no keyframes exist, returns (1, 1).
        """
        keyframes = Keyframe.objects.filter(clip=clip)

        if not keyframes.exists():
            return 1, 1

        distances = []
        for kf in keyframes:
            clip_kf_emb = kf.load_embedding_clip()
            clip_dist = cosine(clip_emb, clip_kf_emb)

            if dino_emb is not None and kf.embedding_dino:
                dino_kf_emb = kf.load_embedding_dino()
                dino_dist = cosine(dino_emb, dino_kf_emb)
                combined = np.mean([clip_dist, dino_dist])
            else:
                combined = clip_dist

            distances.append(combined)

        return min(distances), max(distances)