#!/usr/bin/env python3
"""Rectify EPFL's projective Passageway tracking plane using a depth-fitted metric floor."""
from pathlib import Path
import sys
import json
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.structure import MetricDepthBackend, fit_plane, camera_from_homography
from src.semantics import SemanticBackend
from src.dataset_loader import EPFLDataset
from src.calibrator import EPFLCalibrator
from src.geometry import save_json


def main():
    data = EPFLDataset("data/passageway")
    folder = Path("outputs/passageway/structure")
    folder.mkdir(parents=True, exist_ok=True)
    depth_model = MetricDepthBackend(folder)
    semantic_model = SemanticBackend(folder)
    for camera in data.cameras:
        image = data.frame(camera, 0)
        depth = depth_model.infer(image, camera["id"])
        labels = semantic_model.infer(image, camera["id"])
        if camera["id"] != "cam0":
            continue
        height, width = depth.shape
        focal = 0.9 * width
        k = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
        yy, xx = np.mgrid[:height, :width]
        rays = np.stack([(xx - width / 2) / focal, (yy - height / 2) / focal, np.ones_like(xx)], -1)
        mask = np.isin(labels, [3, 28])
        mask[: height // 2] = False
        n, d, inliers = fit_plane((rays * depth[..., None])[mask][::3], threshold=0.10)
        if n[1] > 0:
            n, d = -n, -d
        forward = np.array([0.0, 0.0, 1.0])
        forward -= n * (forward @ n)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, n)
        right /= np.linalg.norm(right)
        h = np.vstack([-d * right, d * forward, n]) @ np.linalg.inv(k)
        # Use the published projective plane as a common anchor, not as an assumed metric plane.
        import re

        text = (data.root / data.manifest["calibration"]).read_text()
        block = text.split("# Camera 0")[1].split("# Camera")[0]
        source = np.array(
            [
                float(x)
                for x in re.search(r"# Ground plane homography\s*\n([^#]+)", block)[1].split()
            ]
        ).reshape(3, 3)
        source = source @ np.diag(
            [
                data.manifest["calibration_image_size"][0] / width,
                data.manifest["calibration_image_size"][1] / height,
                1.0,
            ]
        )
        rectification = h @ np.linalg.inv(source)
        save_json(
            data.root / "metric_rectification.json",
            {
                "T_published_to_depth_metric": rectification.tolist(),
                "source_camera": "cam0",
                "estimated_K": k.tolist(),
                "floor_plane_camera": [*n.tolist(), d],
                "floor_inlier_ratio": float(inliers.mean()),
                "status": "metric rectification from monocular floor; intrinsics assumed for anchor; not surveyed",
            },
        )
    manifest_path = data.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["plane_rectification_file"] = "metric_rectification.json"
    save_json(manifest_path, manifest)
    data = EPFLDataset(data.root)
    t = EPFLCalibrator().calibrate(data)
    for camera in data.cameras:
        try:
            k, r, tt, c, e = camera_from_homography(t["cameras"][camera["id"]], 360, 288)
            print(camera["id"], "rectified focal", k[0, 0], "constraint_error", e, flush=True)
        except ValueError as error:
            print(camera["id"], str(error), flush=True)


if __name__ == "__main__":
    main()
