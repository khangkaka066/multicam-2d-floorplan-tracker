"""Local ADE20K segmentation for structural labels; cache keyed by image and weights."""

import hashlib
from pathlib import Path
import cv2
import numpy as np
from .geometry import save_json
from .structure import ROOT


class SemanticBackend:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.model = None
        self.model_path = ROOT / "models/segformer-b2"
        self.weights_hash = hashlib.sha256(
            (self.model_path / "pytorch_model.bin").read_bytes()
        ).hexdigest()

    def infer(self, image, name):
        import json

        meta = {
            "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "weights_sha256": self.weights_hash,
        }
        path = self.folder / f"{name}_semantics.npy"
        source = self.folder / f"{name}_semantics_source.json"
        if source.exists() and path.exists() and json.loads(source.read_text()) == meta:
            return np.load(path)
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

        if self.model is None:
            self.processor = SegformerImageProcessor.from_pretrained(
                self.model_path, local_files_only=True
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_path, local_files_only=True
            ).eval()
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model.to(self.device)
        inputs = self.processor(
            images=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits, size=image.shape[:2], mode="bilinear", align_corners=False
            )
            labels = logits.argmax(1)[0].cpu().numpy().astype("uint8")
        np.save(path, labels)
        save_json(source, meta)
        overlay = image.copy()
        for label, color in [(0, (210, 180, 70)), (3, (170, 225, 180)), (14, (40, 100, 240))]:
            overlay[labels == label] = color
        cv2.imwrite(
            str(self.folder / f"{name}_semantics.png"), cv2.addWeighted(image, 0.4, overlay, 0.6, 0)
        )
        print(
            name,
            "semantic pixels",
            {
                "wall": int((labels == 0).sum()),
                "floor": int((labels == 3).sum()),
                "door": int((labels == 14).sum()),
            },
            flush=True,
        )
        return labels
