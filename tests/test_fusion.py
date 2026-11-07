import unittest
from copy import deepcopy
import numpy as np
from src.fusion import robust_fuse, fuse_and_track
from src.bundle_adjustment import refine_homographies
from src.geometry import project


def entry(center):
    return {
        "H_image_to_map": np.eye(3).tolist(),
        "H_map_to_image": np.eye(3).tolist(),
        "camera_center_world": center,
        "camera_xy_estimate": center[:2],
    }


def detection(camera, point, person, index):
    return {
        "camera": camera,
        "uv": list(point),
        "xy": list(point),
        "appearance": [1.0, 0.0] if person == 0 else [0.0, 1.0],
        "method": "ankles",
        "confidence": 0.95,
        "ankle_confidence": 0.95,
        "ankle_censored": False,
        "detection_id": f"{index}:{camera}:{person}",
    }


class FusionTests(unittest.TestCase):
    def test_far_drifting_camera_does_not_pull_near_fusion(self):
        transforms = {
            "cameras": {
                "near0": entry([0, 0, 2]),
                "near1": entry([1, 0, 2]),
                "far": entry([0, 25, 2]),
            }
        }
        group = [
            detection("near0", [0, 0], 0, 0),
            detection("near1", [0.06, 0], 0, 0),
            detection("far", [0, 1.2], 0, 0),
        ]
        point, accepted = robust_fuse(group, transforms)
        self.assertLess(np.linalg.norm(point), 0.04)
        self.assertEqual(len(accepted), 2)
        self.assertFalse(group[2]["fusion_inlier"])
        self.assertGreater(group[0]["fusion_weight"], group[2]["fusion_weight"] * 20)

    def test_close_distinct_people_stay_separate_with_stable_ids(self):
        transforms = {
            "cameras": {
                "cam0": entry([0, -2, 2]),
                "cam1": entry([1, -2, 2]),
                "cam2": entry([-1, -2, 2]),
            }
        }
        records = []
        for index in range(8):
            records.append(
                {
                    "index": index,
                    "detections": [
                        detection(camera, [person * 0.25 + index * 0.01, 0.01 * ci], person, index)
                        for ci, camera in enumerate(transforms["cameras"])
                        for person in range(2)
                    ],
                }
            )
        room = {"id": "r", "polygon": [[-3, -3], [3, -3], [3, 3], [-3, 3]]}
        report = fuse_and_track(records, transforms, room)
        self.assertTrue(all(r["headcounts"]["r"] == 2 for r in records))
        self.assertEqual(
            {p["id"] for p in records[0]["fused_people"]},
            {p["id"] for p in records[-1]["fused_people"]},
        )
        self.assertLess(report["inlier_pair_distance_p95_m"], 0.04)
        for record in records:
            for person in record["fused_people"]:
                self.assertEqual(len(set(person["member_cameras"])), 3)

    def test_short_occlusion_is_flagged_and_not_permanent(self):
        transforms = {"cameras": {"cam0": entry([0, -2, 2]), "cam1": entry([1, -2, 2])}}
        records = []
        for index in range(12):
            detections = [
                detection(c, [p * 0.8, 0], p, index)
                for c in transforms["cameras"]
                for p in range(2)
                if not (p == 1 and 4 <= index <= 10)
            ]
            records.append({"index": index, "detections": detections})
        fuse_and_track(
            records, transforms, {"id": "r", "polygon": [[-3, -3], [3, -3], [3, 3], [-3, 3]]}
        )
        self.assertEqual(records[4]["headcounts"]["r"], 2)
        self.assertTrue(
            any(p["state"] == "predicted_short_occlusion" for p in records[4]["fused_people"])
        )
        self.assertEqual(records[10]["headcounts"]["r"], 1)
        self.assertEqual(records[11]["headcounts"]["r"], 2)

    def test_fusion_uses_pairwise_diameter_not_only_center_radius(self):
        transforms = {
            "cameras": {
                "cam0": entry([0, -2, 2]),
                "cam1": entry([1, -2, 2]),
                "cam2": entry([-1, -2, 2]),
            }
        }
        group = [
            detection("cam0", [-0.30, 0], 0, 0),
            detection("cam1", [0, 0], 0, 0),
            detection("cam2", [0.30, 0], 0, 0),
        ]
        point, accepted = robust_fuse(group, transforms)
        self.assertEqual(len(accepted), 2)
        self.assertLessEqual(np.linalg.norm(np.array(accepted[0]["xy"]) - accepted[1]["xy"]), 0.35)

    def test_projective_bundle_adjustment_on_heldout_motion(self):
        transforms = {
            "cameras": {
                "cam0": entry([-2, -2, 2]),
                "cam1": entry([2, -2, 2]),
                "cam2": entry([0, 2, 2]),
            }
        }
        bad = np.array([[1, 0, 0.15], [0, 1.16, 0.18], [0, 0.05, 1.0]])
        transforms["cameras"]["cam1"]["H_image_to_map"] = bad.tolist()
        transforms["cameras"]["cam1"]["H_map_to_image"] = np.linalg.inv(bad).tolist()
        records = []
        for index in range(70):
            detections = []
            for person in range(2):
                point = np.array(
                    [
                        (-0.7 if person == 0 else 0.7) + 0.2 * np.sin(index * 0.1),
                        -1.6 + index * 0.04 + person * 0.3,
                    ]
                )
                for camera in transforms["cameras"]:
                    d = detection(camera, point, person, index)
                    d["xy"] = project(transforms["cameras"][camera]["H_image_to_map"], [point])[
                        0
                    ].tolist()
                    detections.append(d)
            records.append({"index": index, "detections": detections})
        refined, metrics = refine_homographies(
            records,
            deepcopy(transforms),
            {"id": "r", "polygon": [[-3, -3], [3, -3], [3, 3], [-3, 3]]},
        )
        self.assertTrue(metrics["accepted"])
        self.assertLess(
            metrics["heldout_pair_mean_after_m"], metrics["heldout_pair_mean_before_m"] * 0.6
        )
        self.assertLess(metrics["heldout_pair_p95_after_m"], 0.35)
        self.assertGreater(
            abs(refined["cameras"]["cam1"]["footpoint_bundle_correction"][2][1]), 0.005
        )


if __name__ == "__main__":
    unittest.main()
