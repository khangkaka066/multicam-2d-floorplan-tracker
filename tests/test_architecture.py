import unittest
from pathlib import Path
from copy import deepcopy
import cv2
import numpy as np
from shapely.geometry import Polygon
from src.architecture import room_from_polygon, vectorize_wall_segments, stitch_rooms
from src.structure import fit_plane, camera_from_homography
from src.dataset_loader import EPFLDataset
from src.calibrator import EPFLCalibrator
from src.bev_visualizer import PanelVisualizer


def door(room, wall_index, offset=1.5, width=1.0):
    wall = room["walls"][wall_index]
    a, b = np.array(wall["start"]), np.array(wall["end"])
    t = (b - a) / wall["length_m"]
    d = {
        "id": room["id"] + "_door",
        "wall_id": wall["id"],
        "offset_m": offset,
        "width_m": width,
        "start": (a + t * offset).tolist(),
        "end": (a + t * (offset + width)).tolist(),
    }
    room["doors"].append(d)
    wall["openings"].append(d["id"])
    return d


def fixture():
    a = room_from_polygon("SYNTHETIC_A", [[0, 0], [4, 0], [4, 4], [0, 4]])
    b = room_from_polygon("SYNTHETIC_B", [[0, 0], [3, 0], [3, 2], [2, 2], [2, 4], [0, 4]])
    a["area_m2"] = 16
    b["area_m2"] = 10
    da = door(a, 1)
    db = door(b, 5)
    return [a, b], [{"room_a": a["id"], "door_a": da["id"], "room_b": b["id"], "door_b": db["id"]}]


class ArchitectureTests(unittest.TestCase):
    def test_ransac_plane_rejects_clutter(self):
        rng = np.random.default_rng(2)
        floor = np.c_[rng.uniform(-3, 3, (900, 2)), rng.normal(0, 0.01, 900)]
        clutter = rng.uniform([-3, -3, 0.5], [3, 3, 2], (300, 3))
        n, d, mask = fit_plane(
            np.vstack([floor, clutter]), threshold=0.04, hint=np.array([0, 0, 1])
        )
        self.assertGreater(n[2], 0.99)
        self.assertLess(abs(d), 0.01)
        self.assertGreater(mask[:900].mean(), 0.98)
        self.assertLess(mask[900:].mean(), 0.01)

    def test_l_shape_preserves_reentrant_corner(self):
        points = np.array([[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]])
        segments = list(zip(points, np.roll(points, -1, axis=0)))
        result = vectorize_wall_segments(segments)
        self.assertEqual(len(result), 6)
        self.assertAlmostEqual(Polygon(result).area, 12)
        self.assertFalse(Polygon(result).contains(__import__("shapely").geometry.Point(3, 3)))

    def test_connected_rooms_share_wall_and_door(self):
        rooms, connections = fixture()
        placed, report = stitch_rooms(rooms, connections)
        a, b = placed
        self.assertAlmostEqual(Polygon(a["polygon"]).intersection(Polygon(b["polygon"])).area, 0)
        self.assertTrue(report["shared_walls"])
        np.testing.assert_allclose(a["doors"][0]["start"], b["doors"][0]["end"], atol=1e-8)
        self.assertEqual(a["doors"][0]["connected_room"], b["id"])
        self.assertEqual(len(b["corners"]), 6)

    def test_unconstrained_or_wrong_width_rooms_fail(self):
        rooms, connections = fixture()
        with self.assertRaises(ValueError):
            stitch_rooms(rooms, [])
        bad = deepcopy(rooms)
        bad[1]["doors"][0]["end"][1] -= 0.5
        with self.assertRaises(ValueError):
            stitch_rooms(bad, connections)

    @unittest.skipUnless(
        Path("data/manifest.json").exists(), "Download EPFL samples for this integration test"
    )
    def test_camera_floor_is_in_front_and_above(self):
        data = EPFLDataset("data")
        t = EPFLCalibrator().calibrate(data)
        for camera in data.cameras:
            k, r, translation, c, error = camera_from_homography(
                t["cameras"][camera["id"]], 360, 288
            )
            self.assertGreater(c[2], 0)
            self.assertGreater(translation[2], 0)
            self.assertLess(error, 0.001)
            np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-8)

    def test_outside_projection_is_not_drawn_or_clamped(self):
        class Dataset:
            cameras = [{"id": "cam0"}]

            def frame(self, *args, **kwargs):
                return np.zeros((100, 100, 3), np.uint8)

        room = room_from_polygon("r", [[0, 0], [1, 0], [1, 1], [0, 1]])
        config = {
            "T_map_to_canvas": [[100, 0, 30], [0, 100, 30], [0, 0, 1]],
            "rooms": [room],
            "minimum_display_clearance_m": 0.14,
        }
        detection = {
            "camera": "cam0",
            "room_id": "r",
            "xy": [1.2, 0.8],
            "uv": [20, 20],
            "bbox": [10, 10, 30, 30],
            "method": "ankles",
        }
        record = {"index": 0, "source_frame": 500, "detections": [detection]}
        image = np.full((200, 200, 3), 249, np.uint8)
        output = PanelVisualizer().render(Dataset(), record, image, config)
        self.assertEqual(detection["xy"], [1.2, 0.8])
        np.testing.assert_array_equal(output[110, 200 + 150], image[110, 150])


if __name__ == "__main__":
    unittest.main()
