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


class TopologyTests(unittest.TestCase):
    def rooms(self):
        rooms = [
            room_from_polygon(name, [[0, 0], [4, 0], [4, 4], [0, 4]])
            for name in ["room1", "room2", "room3"]
        ]
        for room, walls in zip(rooms, [[1], [3, 1], [3]]):
            for i, wall in enumerate(walls):
                item = door(room, wall, 1.5, 1.0)
                old_id = item["id"]
                item["id"] = f"D{i+1:02d}"
                room["walls"][wall]["openings"].remove(old_id)
                room["walls"][wall]["openings"].append(item["id"])
        return rooms

    def test_declarative_three_rooms_with_gap(self):
        import json, tempfile
        from src.topology import load_topology
        from src.architecture import layout_rooms

        rooms = self.rooms()
        topology = {
            "links": [
                {"from": "room1:D01", "to": "room2:D01", "gap_m": 0.2},
                {"from": "room2:D02", "to": "room3:D01", "gap_m": 0.0},
            ]
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "topology.json"
            path.write_text(json.dumps(topology))
            placed, report = layout_rooms(rooms, load_topology(path))
        self.assertEqual(report["status"], "stitched")
        self.assertEqual(len(report["connections"]), 2)
        self.assertAlmostEqual(
            min(np.array(placed[1]["polygon"])[:, 0]) - max(np.array(placed[0]["polygon"])[:, 0]),
            0.2,
        )
        np.testing.assert_allclose(placed[1]["doors"][1]["start"], placed[2]["doors"][0]["end"])
        for i, a in enumerate(placed):
            for b in placed[i + 1 :]:
                self.assertLess(
                    Polygon(a["polygon"]).intersection(Polygon(b["polygon"])).area, 1e-8
                )

    def test_trajectory_birth_death_inference_and_three_room_stitch(self):
        from src.topology import detect_entry_exit
        from src.architecture import layout_rooms

        rooms = [
            room_from_polygon(name, [[0, 0], [4, 0], [4, 4], [0, 4]])
            for name in ["room1", "room2", "room3"]
        ]
        segments = {
            "room1": [(12, 20, 3.0, 3.9), (40, 48, 3.0, 3.9)],
            "room2": [
                (23, 31, 0.1, 1.1),
                (51, 59, 0.1, 1.1),
                (70, 78, 3.0, 3.9),
                (100, 108, 3.0, 3.9),
            ],
            "room3": [(81, 89, 0.1, 1.1), (111, 119, 0.1, 1.1)],
        }
        for room in rooms:
            records = []
            for index in range(125):
                detections = []
                for lo, hi, x0, x1 in segments[room["id"]]:
                    if lo <= index <= hi:
                        detections.append(
                            {
                                "camera": "cam",
                                "xy": [x0 + (x1 - x0) * (index - lo) / (hi - lo), 2.0],
                                "appearance": [1.0, 0.0],
                                "appearance_model": "resnet18",
                                "confidence": 0.9,
                                "body_width_m": 0.8,
                            }
                        )
                records.append(
                    {"index": index, "time_seconds": float(index), "detections": detections}
                )
            report = detect_entry_exit(room, records, {"clock_id": "building1"})
            self.assertTrue(report["doors"])
        placed, report = layout_rooms(rooms, {"transition": {"dt_min": 1.0, "dt_max": 5.0}})
        self.assertEqual(report["status"], "stitched")
        self.assertEqual(report["inferred_connection_count"], 2)
        self.assertEqual(len(report["transition_matches"]), 4)
        self.assertEqual(len(report["shared_walls"]), 2)
        self.assertTrue(
            all(
                set(np.rint(np.asarray(r["T_local_to_global"])[:2, :2]).ravel()) <= {-1, 0, 1}
                for r in placed
            )
        )

    def test_independent_recordings_do_not_match_or_snap_by_door_similarity(self):
        from src.architecture import layout_rooms

        rooms = self.rooms()
        for room in rooms:
            for item in room["doors"]:
                item["events"] = [
                    {
                        "kind": "exit" if room["id"] == "room1" else "entry",
                        "time_seconds": 10.0,
                        "track_id": 1,
                        "appearance": [1.0, 0.0],
                        "appearance_model": "resnet18",
                    }
                ]
        placed, report = layout_rooms(rooms)
        self.assertEqual(report["status"], "atlas")
        self.assertFalse(report["connections"])
        self.assertEqual(len(report["components"]), 3)
        self.assertFalse(report["physical_building_registration"])

    def test_time_window_appearance_ambiguity_and_minimum_support(self):
        from src.topology import match_transitions

        rooms = self.rooms()[:2]
        for room in rooms:
            room["doors"] = room["doors"][:1]

        def event(kind, time, ident, feature, clock="shared"):
            return dict(
                kind=kind,
                time_seconds=time,
                track_id=ident,
                appearance=feature,
                appearance_model="resnet18",
                clock_id=clock,
            )

        rooms[0]["doors"][0]["events"] = [
            event("exit", 10, 1, [1, 0]),
            event("exit", 30, 2, [0, 1]),
        ]
        rooms[1]["doors"][0]["events"] = [
            event("entry", 12, 1, [1, 0]),
            event("entry", 32, 2, [0, 1]),
        ]
        links, _ = match_transitions(rooms, dt_min=1, dt_max=5)
        self.assertEqual(len(links), 1)
        rooms[1]["doors"][0]["events"][1]["clock_id"] = "other"
        self.assertFalse(match_transitions(rooms, dt_min=1, dt_max=5)[0])
        rooms[1]["doors"][0]["events"][1]["clock_id"] = "shared"
        rooms[1]["doors"][0]["events"].append(event("entry", 12.1, 4, [1, 0]))
        self.assertFalse(match_transitions(rooms, dt_min=1, dt_max=5)[0])
        with self.assertRaises(ValueError):
            match_transitions(rooms, dt_min=5, dt_max=1)

    def test_explicit_atlas_partial_components_and_invalid_links(self):
        from src.architecture import layout_rooms

        rooms = self.rooms()
        topology = {"links": [{"from": "room1:D01", "to": "room2:D01"}]}
        self.assertEqual(layout_rooms(rooms, topology)[1]["status"], "partial")
        self.assertEqual(layout_rooms(rooms, topology, layout="atlas")[1]["status"], "atlas")
        for link in [
            {"from": "room1:missing", "to": "room2:D01"},
            {"from": "room1:D01", "to": "room2:D01", "gap_m": -0.2},
        ]:
            with self.assertRaises(ValueError):
                layout_rooms(rooms, {"links": [link]})

    def test_rotated_camera_pose_homography_and_bundle_gauges(self):
        from src.scene_atlas import transform_camera

        r = np.diag([1.0, -1.0, -1.0])
        center = np.array([0.2, 1.0, 3.0])
        h = np.array([[0.01, 0, -1], [0, -0.02, 2], [0, 0, 1.0]])
        correction = np.array([[1, 0, 0.1], [0, 1, -0.1], [0, 0, 1.0]])
        entry = {
            "camera_center_world": center.tolist(),
            "camera_xy_estimate": center[:2].tolist(),
            "R_world_to_camera": r.tolist(),
            "t_world_to_camera": (-r @ center).tolist(),
            "optical_axis_world": (r.T @ [0, 0, 1.0]).tolist(),
            "H_image_to_map": (correction @ h).tolist(),
            "H_image_to_map_before_bundle": h.tolist(),
            "footpoint_bundle_correction": correction.tolist(),
        }
        matrix = np.array([[0, -1, 5], [1, 0, 4], [0, 0, 1.0]])
        out = transform_camera(entry, matrix, np.eye(3))
        rotation = np.array(out["R_world_to_camera"])
        translation = np.array(out["t_world_to_camera"])
        np.testing.assert_allclose(out["camera_center_world"], -rotation.T @ translation)
        np.testing.assert_allclose(out["optical_axis_world"], rotation.T @ [0, 0, 1.0])
        np.testing.assert_allclose(
            np.array(out["footpoint_bundle_correction"]) @ out["H_image_to_map_before_bundle"],
            out["H_image_to_map"],
        )
        np.testing.assert_allclose(out["H_image_to_map"], matrix @ correction @ h)

    def test_generic_grouped_manifest_discovery_and_homographies(self):
        import json, tempfile
        from src.dataset_loader import discover_rooms
        from src.calibrator import source_homographies

        with tempfile.TemporaryDirectory() as temp:
            cameras = []
            for room in ["office", "corridor", "studio"]:
                cameras.append(
                    dict(
                        id="ceiling",
                        room_id=room,
                        width=320,
                        height=240,
                        frames=10,
                        fps=25,
                        frames_dir=room,
                        H_image_to_map=[[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]],
                    )
                )
            Path(temp, "manifest.json").write_text(json.dumps({"cameras": cameras}))
            datasets = discover_rooms(temp)
            self.assertEqual(len(datasets), 3)
            self.assertEqual(len({d.manifest["room_id"] for d in datasets}), 3)
            np.testing.assert_allclose(
                source_homographies(datasets[0])["ceiling"], cameras[0]["H_image_to_map"]
            )

    def test_generic_depth_floor_rectification(self):
        from src.boundaries import floor_rectification

        height, width = 100, 120
        k = np.array([[108.0, 0, 60], [0, 108, 50], [0, 0, 1.0]])
        yy, xx = np.mgrid[:height, :width]
        rays = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(k).T
        normal = np.array([0.0, -0.9, -0.4])
        normal /= np.linalg.norm(normal)
        denominator = rays @ normal
        depth = np.divide(
            -1.7, denominator, out=np.zeros_like(denominator), where=abs(denominator) > 1e-8
        )
        labels = np.zeros((height, width), np.uint8)
        labels[60:] = 3
        source = np.array([[0.01, 0, 0.3], [0, 0.02, 0.4], [0.0001, 0, 1.0]])
        report = floor_rectification(
            np.zeros((height, width, 3), np.uint8), depth, labels, source, k
        )
        estimated = np.array(report["floor_plane_camera"])
        self.assertGreater(abs(estimated[:3] @ normal), 0.999)
        self.assertAlmostEqual(abs(estimated[3]), 1.7, places=6)
        self.assertGreater(report["floor_inlier_ratio"], 0.99)
        self.assertTrue(np.isfinite(report["T_published_to_depth_metric"]).all())

    def test_shared_clock_display_uses_offsets_instead_of_frame_indices(self):
        from src.scene_atlas import display_schedule
        from types import SimpleNamespace

        streams = {
            "a": [{"source_frame": 0}, {"source_frame": 1}],
            "b": [{"source_frame": 0}, {"source_frame": 1}],
        }
        datasets = {
            name: SimpleNamespace(
                cameras=[{"fps": 1}],
                manifest={"timing": {"clock_id": "building", "time_offset_seconds": offset}},
            )
            for name, offset in [("a", 0), ("b", 1)]
        }
        schedule, shared = display_schedule(streams, datasets)
        self.assertTrue(shared)
        self.assertEqual(schedule, [{"a": 0, "b": None}, {"a": 1, "b": 0}, {"a": None, "b": 1}])

    def test_build_auto_rectifies_missing_legacy_file(self):
        import tempfile
        from unittest.mock import patch
        from types import SimpleNamespace
        from src.map_builder import prepare_ground_rectification

        height, width = 100, 120
        k = np.array([[108.0, 0, 60], [0, 108, 50], [0, 0, 1.0]])
        yy, xx = np.mgrid[:height, :width]
        rays = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(k).T
        normal = np.array([0.0, -0.9, -0.4])
        normal /= np.linalg.norm(normal)
        denominator = rays @ normal
        depth = np.divide(
            -1.7, denominator, out=np.zeros_like(denominator), where=abs(denominator) > 1e-8
        )
        labels = np.zeros((height, width), np.uint8)
        labels[60:] = 3
        camera = dict(
            id="ceiling",
            room_id="new_corridor",
            width=width,
            height=height,
            H_image_to_map=[[0.01, 0, 0], [0, 0.02, 0], [0, 0, 1.0]],
        )
        with tempfile.TemporaryDirectory() as temp:
            dataset = SimpleNamespace(
                root=Path(temp),
                cameras=[camera],
                manifest={"room_id": "new_corridor", "plane_rectification_file": "missing.json"},
                frame=lambda *args: np.zeros((height, width, 3), np.uint8),
            )
            with patch(
                "src.calibrator.source_homographies",
                return_value={"ceiling": np.asarray(camera["H_image_to_map"])},
            ), patch(
                "src.structure.camera_from_homography", side_effect=ValueError("nonmetric gauge")
            ), patch(
                "src.structure.MetricDepthBackend"
            ) as depth_backend, patch(
                "src.semantics.SemanticBackend"
            ) as semantic_backend:
                depth_backend.return_value.infer.return_value = depth
                semantic_backend.return_value.infer.return_value = labels
                result = prepare_ground_rectification(dataset, Path(temp) / "output")
            self.assertTrue((Path(temp) / "output/structure/ground_rectification.json").exists())
            self.assertIn("ceiling", result["cameras"])
            self.assertIn("ground_rectification.json", dataset.manifest["plane_rectification_file"])

    def test_three_room_assembly_transforms_people_and_saves_verification(self):
        import json, tempfile
        from src.dataset_loader import discover_rooms
        from src.map_builder import configure_canvas
        from src.scene_atlas import assemble
        from src.geometry import project

        rooms = self.rooms()
        # A different door orientation requires a real 90-degree rotation for room3.
        rooms[2] = room_from_polygon("room3", [[0, 0], [4, 0], [4, 4], [0, 4]])
        item = door(rooms[2], 0, 1.5, 1.0)
        old = item["id"]
        item["id"] = "D01"
        rooms[2]["walls"][0]["openings"] = ["D01"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            output = root / "outputs"
            for room in rooms:
                folder = data / room["id"]
                frames = folder / "frames"
                frames.mkdir(parents=True)
                cv2.imwrite(str(frames / "000000.jpg"), np.full((60, 80, 3), 120, np.uint8))
                camera = dict(
                    id="north",
                    room_id=room["id"],
                    width=80,
                    height=60,
                    fps=25,
                    frames=1,
                    frames_dir="frames",
                )
                (folder / "manifest.json").write_text(
                    json.dumps(dict(room_id=room["id"], cameras=[camera]))
                )
                local = output / room["id"]
                local.mkdir(parents=True)
                config = {"rooms": [room], "minimum_display_clearance_m": 0.14}
                config.update(configure_canvas([room]))
                (local / "map_config.json").write_text(json.dumps(config))
                angle = 2.2
                r = np.array(
                    [
                        [1, 0, 0],
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle), np.cos(angle)],
                    ]
                )
                center = np.array([0.1, 0.1, 2.0])
                h = np.diag([0.1, 0.1, 1.0])
                entry = dict(
                    room_id=room["id"],
                    H_image_to_map=h.tolist(),
                    H_map_to_image=np.linalg.inv(h).tolist(),
                    camera_xy_estimate=center[:2].tolist(),
                    camera_center_world=center.tolist(),
                    R_world_to_camera=r.tolist(),
                    t_world_to_camera=(-r @ center).tolist(),
                    optical_axis_world=(r.T @ [0, 0, 1]).tolist(),
                    horizontal_fov_rad=1.0,
                )
                for filename in ["transforms.json", "transforms_refined.json"]:
                    (local / filename).write_text(json.dumps({"cameras": {"north": entry}}))
                detection = dict(
                    camera="north",
                    room_id=room["id"],
                    xy=[2.0, 2.0],
                    uv=[20.0, 20.0],
                    bbox=[10, 10, 30, 30],
                    method="ankles",
                    confidence=0.9,
                    display_accepted=True,
                    detection_id="d0",
                    associated_id=1,
                )
                person = dict(
                    id=1,
                    room_id=room["id"],
                    xy=[2.0, 2.0],
                    member_cameras=["north"],
                    member_detection_ids=["d0"],
                    candidate_detection_ids=["d0"],
                )
                record = dict(
                    index=0,
                    source_frame=0,
                    detections=[detection],
                    fused_people=[person],
                    headcounts={room["id"]: 1},
                )
                (local / "projections.json").write_text(json.dumps([record]))
                for filename in ["verification_metrics.json", "projection_optimization.json"]:
                    (local / filename).write_text("{}")
            path = root / "topology.json"
            path.write_text(
                json.dumps(
                    {
                        "links": [
                            {"from": "room1:D01", "to": "room2:D01", "gap_m": 0.2},
                            {"from": "room2:D02", "to": "room3:D01"},
                        ]
                    }
                )
            )
            config = assemble(output, discover_rooms(data), topology_path=path)
            self.assertTrue(config["physical_building_registration"])
            self.assertIsNotNone(cv2.imread(str(output / "verification_bev.png")))
            records = json.loads((output / "projections.json").read_text())
            for person in records[0]["fused_people"]:
                matrix = config["scene_transforms"][person["room_id"]]
                np.testing.assert_allclose(person["xy"], project(matrix, [[2.0, 2.0]])[0])
            third = np.asarray(config["scene_transforms"]["room3"])
            self.assertAlmostEqual(abs(third[0, 1]), 1.0)
            self.assertEqual(len(records[0]["fused_people"]), 3)

    def test_homography_validity_is_independent_of_homogeneous_scale(self):
        from src.calibrator import source_homographies
        from types import SimpleNamespace

        h = np.array([[0.01, 0, 0.3], [0, 0.02, 0.4], [0.0001, 0, 1.0]])
        for factor in [1.0, 1e-12, 1e12]:
            camera = {
                "id": "camera",
                "width": 320,
                "height": 240,
                "H_image_to_map": (h * factor).tolist(),
            }
            dataset = SimpleNamespace(manifest={}, cameras=[camera])
            result = source_homographies(dataset)["camera"]
            np.testing.assert_allclose(result / result[2, 2], h)

    def test_inferred_body_envelopes_align_without_scaling_rooms(self):
        from src.architecture import layout_rooms

        rooms = self.rooms()[:2]
        for room in rooms:
            room["doors"] = room["doors"][:1]
        opening = rooms[1]["doors"][0]
        wall = rooms[1]["walls"][3]
        opening["end"][1] -= 0.2
        opening["width_m"] = 1.2
        for room in rooms:
            kind = "exit" if room["id"] == "room1" else "entry"
            room["doors"][0]["events"] = [
                dict(
                    kind=kind,
                    time_seconds=t + (2 if kind == "entry" else 0),
                    track_id=i,
                    appearance=[1.0, 0.0],
                    appearance_model="resnet18",
                    clock_id="shared",
                )
                for i, t in enumerate([10.0, 30.0])
            ]
        topology = {
            "links": [{"from": "room1:D01", "to": "room2:D01", "gap_m": 0.2}],
            "transition": {"dt_min": 1.0, "dt_max": 5.0},
        }
        placed, report = layout_rooms(rooms, topology)
        self.assertEqual(report["status"], "stitched")
        self.assertEqual(report["connections"][0]["gap_m"], 0.2)
        self.assertIn("original_door_widths_m", report["connections"][0])
        for room in placed:
            self.assertAlmostEqual(Polygon(room["polygon"]).area, 16.0)
        self.assertAlmostEqual(placed[0]["doors"][0]["width_m"], placed[1]["doors"][0]["width_m"])


if __name__ == "__main__":
    unittest.main()
