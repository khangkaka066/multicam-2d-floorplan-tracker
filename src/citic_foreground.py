"""Temporal image evidence suppresses pose detections on static showroom exhibits."""
import cv2
import numpy as np


def pose_supported(detection):
    """Large extrapolations require stronger evidence than visible ankle contacts."""
    if detection["method"] in ["shoulders_x5", "elbows_x3"]:
        head = np.asarray(detection["keypoints"])[:5, 2]
        return bool(detection["detector_confidence"] >= 0.65 and (head >= 0.5).sum() >= 2)
    return True


class TemporalForeground:
    def __init__(self, dataset, output):
        self.backgrounds = {}
        folder = output / "temporal_backgrounds"
        folder.mkdir(exist_ok=True)
        for camera in dataset.cameras:
            path = folder / (camera["id"] + ".png")
            if path.exists():
                background = cv2.imread(str(path))
            else:
                frames = [cv2.resize(dataset.frame(camera, int(i)), (480, 270))
                          for i in np.linspace(0, camera["frames"] - 2, 13)]
                background = np.median(frames, axis=0).astype(np.uint8)
                if not cv2.imwrite(str(path), background):
                    raise RuntimeError(f"Cannot save background {path}")
            self.backgrounds[camera["id"]] = background

    def score(self, frame, camera, detections):
        background = self.backgrounds[camera["id"]]
        small = cv2.resize(frame, (background.shape[1], background.shape[0]))
        difference = cv2.absdiff(cv2.GaussianBlur(small, (3, 3), 0),
                                 cv2.GaussianBlur(background, (3, 3), 0))
        mask = difference.max(axis=2) > 30
        for d in detections:
            a, b, c, e = d["bbox"]
            # Central body crop reduces background/window influence around narrow people.
            a, c = a + 0.15 * (c - a), c - 0.15 * (c - a)
            x1, x2 = np.int32(np.array([a, c]) * mask.shape[1] / frame.shape[1])
            y1, y2 = np.int32(np.array([b, e]) * mask.shape[0] / frame.shape[0])
            crop = mask[max(0, y1):min(mask.shape[0], y2), max(0, x1):min(mask.shape[1], x2)]
            d["foreground_fraction"] = float(crop.mean()) if crop.size else 0.
            d["foreground_supported"] = d["foreground_fraction"] >= 0.12


class FloorSupport:
    """Reject explicit furniture/mirror contacts; preserve uncertain corridor labels."""
    def __init__(self, dataset, output):
        from .semantics import SemanticBackend

        folder = output / "temporal_backgrounds"
        backend = SemanticBackend(folder)
        self.labels = {}
        for camera in dataset.cameras:
            background = cv2.imread(str(folder / (camera["id"] + ".png")))
            if background is None:
                raise ValueError("Prepare temporal backgrounds before semantic floor support")
            self.labels[camera["id"]] = backend.infer(background, camera["id"] + "_background")

    def score(self, detection, camera):
        u, v = detection["uv"]
        if not (0 <= u < camera["width"] and 0 <= v < camera["height"]):
            detection.update(floor_supported=True, floor_support_fraction=None,
                             floor_support_status="outside_image; extrapolation retained")
            return
        labels = self.labels[camera["id"]]
        x, y = np.int32(np.array([u, v]) * [labels.shape[1] / camera["width"],
                                           labels.shape[0] / camera["height"]])
        crop = labels[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        # Bright corridor floors can be mislabeled wall/door. Those classes are ambiguous,
        # whereas a contact inside a cabinet, mirror or shelving unit is positive rejection evidence.
        furniture = [7, 10, 15, 19, 23, 24, 33, 45]
        support = 1. - float(np.isin(crop, furniture).mean())
        detection.update(floor_supported=support >= 0.5, floor_support_fraction=support,
                         semantic_floor_fraction=float(np.isin(crop, [3, 28]).mean()),
                         semantic_contact_label=int(labels[y, x]),
                         floor_support_status="explicit furniture/mirror exclusion; ambiguous background labels retained")
