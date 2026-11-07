import re
from typing import Protocol
import numpy as np
from .geometry import project


class Calibrator(Protocol):
    def calibrate(self, dataset) -> dict: ...


class EPFLCalibrator:
    def calibrate(self, dataset):
        text = (dataset.root / dataset.manifest["calibration"]).read_text()
        matrices = {}
        import json

        rectification = np.eye(3)
        if dataset.manifest.get("plane_rectification_file"):
            rectification = np.array(
                json.loads(
                    (dataset.root / dataset.manifest["plane_rectification_file"]).read_text()
                )["T_published_to_depth_metric"]
            )
        for camera in dataset.cameras:
            number = int(camera["id"].removeprefix("cam"))
            block = re.search(rf"# Camera {number}\b(.*?)(?=# Camera|\Z)", text, re.S).group(1)
            match = re.search(r"# Ground plane homography\s*\n([^#]+)", block)
            matrix = np.array([float(x) for x in match.group(1).split()]).reshape(3, 3)
            source_width, source_height = dataset.manifest["calibration_image_size"]
            matrix = (
                rectification
                @ matrix
                @ np.diag([source_width / camera["width"], source_height / camera["height"], 1])
            )
            matrix /= matrix[2, 2]
            if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-10:
                raise ValueError(f'Singular calibration for {camera["id"]}')
            matrices[camera["id"]] = matrix
        # H alone does not identify Euclidean camera centers. Use vertical VP at infinity,
        # explicitly an approximation (zero roll, approximately vertical image axes).
        estimates = np.array([project(h, [[180, 1e8]])[0] for h in matrices.values()])
        centroid = estimates.mean(axis=0)
        canonical = np.eye(3)
        canonical[:2, 2] = -centroid
        result = {
            "schema_version": 1,
            "units": dataset.manifest["coordinate_units"],
            "source": "EPFL published image-to-topview ground homographies; not reconstructed poses",
            "convention": "H_map_to_image maps canonical floor to image; inverse maps image to floor",
            "gauge": {
                "origin_epfl": centroid.tolist(),
                "method": "centroid of approximate camera ground positions",
            },
            "cameras": {},
        }
        for camera, position in zip(dataset.cameras, estimates):
            h = canonical @ matrices[camera["id"]]
            h /= h[2, 2]
            points = [[80, 160], [180, 250], [280, 200]]
            error = np.linalg.norm(
                project(np.linalg.inv(h), project(h, points)) - points, axis=1
            ).max()
            result["cameras"][camera["id"]] = {
                "room_id": camera["room_id"],
                "H_image_to_map": h.tolist(),
                "H_map_to_image": np.linalg.inv(h).tolist(),
                "camera_xy_estimate": (position - centroid).tolist(),
                "camera_position_status": "approximate vertical-vanishing-point assumption; not measured extrinsics",
                "roundtrip_error_px": float(error),
            }
        return result


def floor_alignment_report(dataset, transforms):
    """Independent static feature consistency, not a ground-truth accuracy claim."""
    import cv2
    from itertools import combinations

    sift = cv2.SIFT_create(nfeatures=2000)
    features = {}
    for camera in dataset.cameras:
        image = dataset.frame(camera, 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 30), (180, 210, 210))
        mask[: int(image.shape[0] * 0.4)] = 0
        features[camera["id"]] = sift.detectAndCompute(image, mask)
    report = {
        "method": "SIFT ratio 0.7 + pairwise RANSAC (3px); compare published ground homographies",
        "limitation": "Heuristic floor mask can include furniture. This measures consistency, not wall or metric accuracy.",
        "pairs": [],
    }
    matcher = cv2.BFMatcher()
    for a, b in combinations(features, 2):
        ka, da = features[a]
        kb, db = features[b]
        matches = (
            [
                pair[0]
                for pair in matcher.knnMatch(da, db, k=2)
                if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
            ]
            if da is not None and db is not None
            else []
        )
        result = {"cameras": [a, b], "ratio_matches": len(matches)}
        if len(matches) >= 6:
            pa = np.float32([ka[m.queryIdx].pt for m in matches])
            pb = np.float32([kb[m.trainIdx].pt for m in matches])
            h, inliers = cv2.findHomography(pa, pb, cv2.RANSAC, 3)
            if h is not None:
                valid = inliers.ravel().astype(bool)
                error = np.linalg.norm(
                    project(transforms["cameras"][a]["H_image_to_map"], pa[valid])
                    - project(transforms["cameras"][b]["H_image_to_map"], pb[valid]),
                    axis=1,
                )
                result.update(
                    {
                        "ransac_inliers": int(valid.sum()),
                        "median_alignment_epfl_units": float(np.median(error)),
                        "p90_alignment_epfl_units": float(np.percentile(error, 90)),
                        "median_pair_reprojection_px": float(
                            np.median(np.linalg.norm(project(h, pa[valid]) - pb[valid], axis=1))
                        ),
                    }
                )
        report["pairs"].append(result)
    return report
