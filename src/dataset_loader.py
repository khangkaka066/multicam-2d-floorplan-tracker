"""Manifest-based room discovery with image/video adapters and an EPFL compatibility alias."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Protocol
import cv2


class DatasetLoader(Protocol):
    cameras: list

    def frame(self, camera: dict, index: int, verification: bool = False): ...


class RoomDataset:
    def __init__(self, root, room_id=None):
        self.root = Path(root).resolve()
        self.manifest = deepcopy(json.loads((self.root / "manifest.json").read_text()))
        cameras = self.manifest.get("cameras", [])
        ids = {c.get("room_id", self.manifest.get("room_id")) for c in cameras}
        if room_id is None:
            if len(ids) != 1 or None in ids:
                raise ValueError("Select a room from a multi-room manifest using discover_rooms()")
            room_id = next(iter(ids))
        if not room_id or any(c in str(room_id) for c in "/\\:") or room_id in [".", ".."]:
            raise ValueError("Room IDs must be safe directory names without colons")
        self.manifest["room_id"] = room_id
        self.manifest.setdefault("source", "local manifest")
        self.cameras = []
        for original in cameras:
            if original.get("room_id", self.manifest.get("room_id")) != room_id:
                continue
            c = deepcopy(original)
            c["room_id"] = room_id
            if any(x in c["id"] for x in "/\\:"):
                raise ValueError("Camera IDs must not contain path separators or colons")
            c.setdefault("verification_frames_dir", c.get("frames_dir"))
            c.setdefault("verification_video", c.get("video"))
            c.setdefault("verification_start_frame", self.manifest.get("first_source_frame", 0))
            if not all(k in c for k in ["fps", "width", "height", "frames"]):
                image = self.frame(c, 0)
                c.setdefault("width", image.shape[1])
                c.setdefault("height", image.shape[0])
                cap = cv2.VideoCapture(str(self.root / c["video"])) if c.get("video") else None
                c.setdefault(
                    "fps", cap.get(cv2.CAP_PROP_FPS) if cap else self.manifest.get("fps", 25)
                )
                c.setdefault(
                    "frames",
                    (
                        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if cap
                        else len(list((self.root / c["frames_dir"]).glob("*")))
                    ),
                )
                if cap:
                    cap.release()
            if c["fps"] <= 0 or c["frames"] < 1:
                raise ValueError("Camera fps and frame count must be positive")
            self.cameras.append(c)
        if not self.cameras or len({c["id"] for c in self.cameras}) != len(self.cameras):
            raise ValueError("Empty or duplicate camera IDs")
        if len({c["fps"] for c in self.cameras}) != 1:
            raise ValueError("Synchronize/resample intra-room camera rates before importing")
        self.manifest["cameras"] = self.cameras

    def frame(self, camera, index, verification=False):
        key = "verification_frames_dir" if verification else "frames_dir"
        folder = camera.get(key)
        if folder:
            pattern = camera.get("frame_pattern", "{index:06d}.jpg")
            path = self.root / folder / pattern.format(index=index)
            frame = cv2.imread(str(path))
        else:
            key = "verification_video" if verification else "video"
            path = self.root / camera[key]
            cap = cv2.VideoCapture(str(path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                frame = None
        if frame is None:
            raise FileNotFoundError(f"{path}: frame {index}")
        return frame

    def observation_fingerprint(self, camera, count):
        import hashlib

        digest = hashlib.sha256()
        folder = camera.get("verification_frames_dir")
        if folder:
            for index in range(count):
                path = (
                    self.root
                    / folder
                    / camera.get("frame_pattern", "{index:06d}.jpg").format(index=index)
                )
                digest.update(path.read_bytes())
        else:
            with (self.root / camera["verification_video"]).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    def profile(self):
        return {
            "source": self.manifest["source"],
            "rooms": [self.manifest["room_id"]],
            "camera_count": len(self.cameras),
            "cameras": self.cameras,
            "calibration_available": bool(
                self.manifest.get("calibration")
                or self.manifest.get("homographies")
                or all("H_image_to_map" in c for c in self.cameras)
            ),
            "intrinsics_available": all("K" in c for c in self.cameras),
            "wall_ground_truth_available": False,
        }


EPFLDataset = RoomDataset


def discover_rooms(root):
    """Discover room manifests recursively, including multiple rooms in one camera manifest."""
    root = Path(root).resolve()
    paths = sorted(root.rglob("manifest.json"))
    rooms = []
    for path in paths:
        manifest = json.loads(path.read_text())
        cameras = manifest.get("cameras", [])
        if not cameras:
            continue
        ids = {c.get("room_id", manifest.get("room_id")) for c in cameras}
        for room_id in sorted(ids, key=str):
            rooms.append(RoomDataset(path.parent, room_id))
    if not rooms:
        raise ValueError(f"No camera manifest.json found under {root}")
    ids = [d.manifest["room_id"] for d in rooms]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate room IDs across manifests")
    return rooms
