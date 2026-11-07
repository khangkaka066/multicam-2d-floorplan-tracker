import json
from pathlib import Path
from typing import Protocol
import cv2


class DatasetLoader(Protocol):
    cameras: list

    def frame(self, camera: dict, index: int, verification: bool = False): ...


class EPFLDataset:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.cameras = self.manifest["cameras"]
        if not self.cameras:
            raise ValueError("Empty camera manifest")
        if len({c["room_id"] for c in self.cameras}) != 1:
            raise ValueError(
                "EPFL sample adapter supports one room; disjoint rooms require alignment constraints"
            )
        if len({c["fps"] for c in self.cameras}) != 1:
            raise ValueError("Unsynchronized frame rates")

    def frame(self, camera, index, verification=False):
        key = "verification_frames_dir" if verification else "frames_dir"
        path = self.root / camera[key] / f"{index:06d}.jpg"
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(path)
        return frame

    def profile(self):
        return {
            "source": self.manifest["source"],
            "rooms": sorted({c["room_id"] for c in self.cameras}),
            "camera_count": len(self.cameras),
            "cameras": self.cameras,
            "calibration_available": True,
            "intrinsics_available": False,
            "wall_ground_truth_available": False,
        }
