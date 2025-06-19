import torch
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel
import timm
from PIL import Image
import numpy as np
from VideoSearch.utils.hardware import EmbeddingModelSelector

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
            return {
                "clip": clip_emb,
                "dino": None
            }

        dino_emb = self.get_dino_embedding(image)
        return {
            "clip": clip_emb,
            "dino": dino_emb
        }