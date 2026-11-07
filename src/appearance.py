"""Appearance descriptors from pretrained image features, not claimed as a trained ReID model."""

from pathlib import Path
import cv2
import numpy as np
import torch


class AppearanceEncoder:
    def __init__(self, weights=None):
        from torchvision.models import resnet18

        path = Path(weights or Path(__file__).resolve().parents[1] / "models/resnet18-f37072fd.pth")
        self.model = resnet18(weights=None)
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.model.fc = torch.nn.Identity()
        self.model.eval()

    def encode(self, frame, detections):
        batch = []
        for d in detections:
            a, b, c, e = np.rint(d["bbox"]).astype(int)
            crop = frame[max(0, b) : min(frame.shape[0], e), max(0, a) : min(frame.shape[1], c)]
            if not crop.size:
                raise ValueError("Empty appearance crop")
            rgb = (
                cv2.cvtColor(cv2.resize(crop, (112, 224)), cv2.COLOR_BGR2RGB).astype("float32")
                / 255
            )
            rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
                [0.229, 0.224, 0.225], np.float32
            )
            batch.append(np.transpose(rgb, (2, 0, 1)))
        if not batch:
            return
        with torch.inference_mode():
            features = self.model(torch.from_numpy(np.stack(batch))).numpy()
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)
        for detection, feature in zip(detections, features):
            detection["color_appearance"] = detection.get(
                "clothing_appearance", detection["appearance"]
            )
            detection["appearance"] = feature.tolist()
            detection["appearance_model"] = (
                "ImageNet ResNet18 crop descriptor; not a dedicated ReID network"
            )
