from typing import Protocol
import cv2
import numpy as np


class PersonLocator(Protocol):
    def detect(self, frame) -> list: ...


def ground_point(keypoints, bbox, threshold=0.5):
    """AGENTS ratios are total head-to-foot / visible-head-to-joint height."""
    k = np.asarray(keypoints)
    visible = lambda ids: [i for i in ids if k[i, 2] >= threshold]
    ankles = visible([15, 16])
    if ankles:
        point = k[ankles, :2].mean(0)
        point[1] += 0.02 * (bbox[3] - bbox[1])
        return point, "ankles", float(k[ankles, 2].mean())
    heads = visible([0, 1, 2, 3, 4])
    if not heads:
        return None
    top = k[heads, :2].mean(0)
    for ids, factor, label in [
        ([13, 14], 1.6, "knees_x1.6"),
        ([11, 12], 2.0, "hips_x2"),
        ([7, 8], 3.0, "elbows_x3"),
        ([5, 6], 5.0, "shoulders_x5"),
    ]:
        indices = visible(ids)
        if indices:
            joint = k[indices, :2].mean(0)
            if joint[1] - top[1] < 3:
                return None
            return (
                np.array([joint[0], top[1] + factor * (joint[1] - top[1])]),
                label,
                float(k[indices, 2].mean()) / factor,
            )
    return None


class YOLOPoseLocator:
    def __init__(self, weights, device="cpu", imgsz=640, conf=0.25):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device
        self.imgsz = imgsz
        self.conf = conf

    def detect(self, frame):
        result = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
        )[0]
        detections = []
        if result.keypoints is None:
            return detections
        for keypoints, box, score in zip(
            result.keypoints.data.cpu().numpy(),
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            point = ground_point(keypoints, box)
            if point is None:
                continue
            x1, y1, x2, y2 = np.rint(box).astype(int)
            crop = frame[max(0, y1) : min(frame.shape[0], y2), max(0, x1) : min(frame.shape[1], x2)]
            if not crop.size:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # Value and vertical clothing bands distinguish dark trousers, light trousers and jackets.
            descriptors = []
            for lo, hi in [(0.15, 0.55), (0.55, 0.95)]:
                band = hsv[int(len(hsv) * lo) : max(int(len(hsv) * hi), int(len(hsv) * lo) + 1)]
                histogram = cv2.calcHist(
                    [band], [0, 1, 2], None, [8, 4, 8], [0, 180, 0, 256, 0, 256]
                ).flatten()
                histogram /= max(np.linalg.norm(histogram), 1e-6)
                descriptors.extend(histogram.tolist())
            clothing = np.asarray(descriptors)
            clothing /= max(np.linalg.norm(clothing), 1e-6)
            histogram = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).flatten()
            histogram /= max(np.linalg.norm(histogram), 1e-6)
            visible_ankles = keypoints[[15, 16], 2] >= 0.5
            ankle_confidence = (
                float(keypoints[[15, 16], 2][visible_ankles].mean())
                if visible_ankles.any()
                else 0.0
            )
            ankle_center = (
                keypoints[[15, 16], :2][visible_ankles].mean(0).tolist()
                if visible_ankles.any()
                else None
            )
            detections.append(
                {
                    "uv": point[0].tolist(),
                    "method": point[1],
                    "confidence": point[2],
                    "bbox": box.tolist(),
                    "appearance": histogram.tolist(),
                    "clothing_appearance": clothing.tolist(),
                    "ankle_confidence": ankle_confidence,
                    "ankle_uv": ankle_center,
                    "detector_confidence": float(score),
                    "keypoints": keypoints.tolist(),
                    "inference_imgsz": self.imgsz,
                }
            )
        return detections


class MultiScalePoseLocator:
    """Recover far subjects at high resolution while retaining cropped near subjects at small scales."""

    def __init__(self, weights, scales, device="cpu", conf=0.25):
        self.detector = YOLOPoseLocator(weights, device=device, imgsz=max(scales), conf=conf)
        self.scales = scales
        from .appearance import AppearanceEncoder

        self.encoder = AppearanceEncoder()

    @staticmethod
    def overlap(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
        aa = (ax2 - ax1) * (ay2 - ay1)
        bb = (bx2 - bx1) * (by2 - by1)
        return intersection / max(aa + bb - intersection, 1e-6), intersection / max(
            min(aa, bb), 1e-6
        )

    def detect(self, frame):
        candidates = []
        for scale in self.scales:
            self.detector.imgsz = scale
            for d in self.detector.detect(frame):
                d["scale_sources"] = [scale]
                k = np.asarray(d["keypoints"])
                visible = k[[15, 16], 2] >= 0.5
                d["ankle_censored"] = bool(
                    visible.any()
                    and (
                        k[[15, 16], 1][visible].max() >= frame.shape[0] - 6
                        or k[[15, 16], 0][visible].min() < 2
                        or k[[15, 16], 0][visible].max() > frame.shape[1] - 3
                    )
                )
                candidates.append(d)

        def quality(d):
            return (
                d["ankle_confidence"] * (0.4 if d["ankle_censored"] else 1)
                + 0.2 * d["detector_confidence"]
                + 0.1 * d["confidence"]
            )

        merged = []
        for d in sorted(candidates, key=quality, reverse=True):
            match = next(
                (
                    m
                    for m in merged
                    if self.overlap(d["bbox"], m["bbox"])[0] > 0.35
                    or self.overlap(d["bbox"], m["bbox"])[1] > 0.80
                ),
                None,
            )
            if match is None:
                merged.append(d)
            else:
                match["scale_sources"] = sorted(set(match["scale_sources"] + d["scale_sources"]))
        self.encoder.encode(frame, merged)
        return merged
