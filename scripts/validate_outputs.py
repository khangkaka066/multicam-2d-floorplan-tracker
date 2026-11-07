#!/usr/bin/env python3
"""Check real-scene artifacts, pose identities, wall-reaching rays and unmodified projections."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from shapely.geometry import Polygon, Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.geometry import project, save_json
from src.boundaries import view_polygon


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs"
    config = json.loads((output / "map_config.json").read_text())
    transforms = json.loads((output / "transforms_refined.json").read_text())
    assert config["schema_version"] == 3 and len(config["rooms"]) == 2
    assert config["physical_building_registration"] is False
    assert not config["alignment"]["connections"] and not config["alignment"]["shared_walls"]
    jpeg_count = 0
    for folder in [root / "data", root / "data/passageway"]:
        manifest = json.loads((folder / "manifest.json").read_text())
        for camera in manifest["cameras"]:
            for key in ["frames_dir", "verification_frames_dir"]:
                files = sorted((folder / camera[key]).glob("*.jpg"))
                assert len(files) == 100
                for path in files:
                    image = cv2.imread(str(path))
                    assert image is not None and image.shape[:2] == (
                        camera["height"],
                        camera["width"],
                    )
                    jpeg_count += 1
    max_corner_error = 0.0
    for room in config["rooms"]:
        p = np.array(room["polygon"])
        assert Polygon(p).is_valid
        edges = np.roll(p, -1, axis=0) - p
        for a, b in zip(edges, np.roll(edges, -1, axis=0)):
            error = abs(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            max_corner_error = max(max_corner_error, float(error))
            assert error < 1e-8
        for door in room["doors"]:
            assert door.get("supporting_tracklets", 0) >= 2
            wall = next(w for w in room["walls"] if w["id"] == door["wall_id"])
            a, b = np.array(wall["start"]), np.array(wall["end"])
            u = (b - a) / wall["length_m"]
            np.testing.assert_allclose(door["start"], a + u * door["offset_m"], atol=1e-8)
            np.testing.assert_allclose(
                door["end"], a + u * (door["offset_m"] + door["width_m"]), atol=1e-8
            )
    mount_distances = {}
    fov_angles = {}
    max_pose_error = 0.0
    max_ray_range = 0.0
    for camera, e in transforms["cameras"].items():
        r = np.array(e["R_world_to_camera"])
        t = np.array(e["t_world_to_camera"])
        c = np.array(e["camera_center_world"])
        d = np.array(e["optical_axis_world"])
        max_pose_error = max(max_pose_error, float(np.linalg.norm(c + r.T @ t)))
        np.testing.assert_allclose(c, -r.T @ t, atol=1e-8)
        np.testing.assert_allclose(d, r.T @ [0, 0, 1], atol=1e-8)
        room = next(v for v in config["rooms"] if v["id"] == e["room_id"])
        shape = Polygon(room["polygon"])
        distance = shape.boundary.distance(Point(c[:2]))
        assert distance < 0.3 and shape.contains(Point(c[:2]))
        mount_distances[camera] = distance
        rays = view_polygon(c[:2], d, e["horizontal_fov_rad"], room["polygon"])
        for point in rays[1:]:
            assert shape.boundary.distance(Point(point)) < 1e-6
        max_ray_range = max(max_ray_range, float(np.linalg.norm(rays[1:] - c[:2], axis=1).max()))
        fov_angles[camera] = float(np.rad2deg(e["horizontal_fov_rad"]))
        h = np.array(e["H_image_to_map"])
        hi = np.array(e["H_map_to_image"])
        np.testing.assert_allclose(h @ hi, np.eye(3), atol=1e-8)
        np.testing.assert_allclose(
            e["H_image_to_canvas"], np.asarray(config["T_map_to_canvas"]) @ h, atol=1e-8
        )
        if "footpoint_bundle_correction" in e:
            np.testing.assert_allclose(
                np.asarray(e["footpoint_bundle_correction"]) @ e["H_image_to_map_before_bundle"],
                h,
                atol=1e-8,
            )
    records = json.loads((output / "projections.json").read_text())
    max_projection_error = 0.0
    displayed = 0
    rejected = 0
    for record in records:
        for detection in record["detections"]:
            e = transforms["cameras"][detection["camera"]]
            xy = project(e["H_image_to_map"], [detection["uv"]])[0]
            max_projection_error = max(
                max_projection_error, float(np.linalg.norm(xy - detection["xy"]))
            )
            polygon = next(r["polygon"] for r in config["rooms"] if r["id"] == detection["room_id"])
            clearance = float(cv2.pointPolygonTest(np.array(polygon, np.float32), tuple(xy), True))
            assert bool(detection["display_accepted"]) == (
                clearance >= config["minimum_display_clearance_m"]
            )
            if detection["display_accepted"]:
                assert clearance > 0
                displayed += 1
            else:
                rejected += 1
    assert max_projection_error < 1e-7
    consensus_diameters = []
    prediction_count = 0
    for record in records:
        by_id = {d["detection_id"]: d for d in record["detections"]}
        people = record["fused_people"]
        assert len({p["id"] for p in people}) == len(people)
        for room in config["rooms"]:
            assert record["headcounts"][room["id"]] == sum(
                p["room_id"] == room["id"] for p in people
            )
        for person in people:
            polygon = next(r["polygon"] for r in config["rooms"] if r["id"] == person["room_id"])
            assert (
                cv2.pointPolygonTest(np.asarray(polygon, np.float32), tuple(person["xy"]), True)
                >= config["minimum_display_clearance_m"]
            )
            if person["state"] == "predicted_short_occlusion":
                prediction_count += 1
                assert person["predicted_frame_gap"] <= 5
                continue
            members = [by_id[i] for i in person["member_detection_ids"]]
            assert len(members) >= 2 and len({d["camera"] for d in members}) == len(members)
            positions = np.array([d["xy"] for d in members])
            weights = np.array([d["fusion_weight"] for d in members])
            np.testing.assert_allclose(
                person["xy"], np.average(positions, axis=0, weights=weights), atol=1e-8
            )
            diameter = float(np.linalg.norm(positions[:, None] - positions[None, :], axis=2).max())
            assert diameter <= 0.350001
            consensus_diameters.append(diameter)
    selected = next(r for r in records if r["index"] == 78)
    # User-specified reference labels are test expectations, never an inference input.
    assert selected["headcounts"] == {"laboratory": 2, "passageway": 4}, selected["headcounts"]
    assert len(selected["fused_people"]) == 6 and all(
        p["state"] == "observed" for p in selected["fused_people"]
    )
    assert len([d for d in selected["detections"] if d["camera"] == "LAB/cam2"]) == 2
    for camera in ["PASS/cam2", "PASS/cam3"]:
        assert len([d for d in selected["detections"] if d["camera"] == camera]) == 4
    raw_validation_target_met = []
    for scene in ["laboratory", "passageway"]:
        metric = json.loads((output / scene / "projection_optimization.json").read_text())
        raw_validation_target_met.append(metric["target_met_for_all_heldout_pairs"])
        assert metric["requested_frame_78_used_for_fit"] is False
        assert metric["validation_pixels_used_for_boundary_loss"] is False
    video = cv2.VideoCapture(str(output / "verification_projection.mp4"))
    count = 0
    while True:
        ok, _ = video.read()
        if not ok:
            break
        count += 1
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    assert count == len(records) == 100 and fps == 25
    for name in ["global_2d_map.png", "verification_bev.png"]:
        assert cv2.imread(str(output / name)) is not None
    report = {
        "passed": True,
        "real_scenes": 2,
        "real_cameras": len(transforms["cameras"]),
        "decoded_jpeg_frames": jpeg_count,
        "decoded_verification_video_frames": count,
        "displayed_points_strictly_inside": displayed,
        "rejected_unmodified_projections": rejected,
        "max_projection_error_m": max_projection_error,
        "inspected_headcounts": selected["headcounts"],
        "inspected_global_ids": [p["id"] for p in selected["fused_people"]],
        "maximum_accepted_fusion_diameter_m": max(consensus_diameters),
        "predicted_person_observations": prediction_count,
        "max_group_diameter_at_requested_frame_m": max(
            p["max_pairwise_distance_m"] for p in selected["fused_people"]
        ),
        "all_raw_validation_pairs_meet_035m": all(raw_validation_target_met),
        "max_C_identity_error_m": max_pose_error,
        "orthogonal_corner_max_abs_cosine": max_corner_error,
        "camera_distance_to_perimeter_m": mount_distances,
        "estimated_horizontal_fov_degrees": fov_angles,
        "maximum_fov_ray_length_m": max_ray_range,
        "shared_doorway_registration": "UNKNOWN; not fabricated",
        "scope": "Internal geometry and artifact consistency; dimensions, priors and trajectory doors are not survey validation.",
    }
    save_json(output / "artifact_validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
