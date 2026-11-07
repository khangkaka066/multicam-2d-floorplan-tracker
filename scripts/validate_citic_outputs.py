#!/usr/bin/env python3
"""Verify real CITIC artifacts, independent geometry identities and measured quality gates."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from shapely.geometry import Point, Polygon, shape
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.citic_dataset import CiticDataset, build_calibration
from src.citic_evaluation import evaluate, evaluate_transitions
from src.geometry import project, save_json
from src.person_locator import ground_point
from src.citic_foreground import pose_supported
from src.citic_visualizer import camera_fov_mask, camera_label_layout, _overlap


def validate(output, data):
    output, data = Path(output), Path(data)
    dataset = CiticDataset(data)
    config = json.loads((output / "map_config.json").read_text())
    transforms = json.loads((output / "transforms_refined.json").read_text())
    base = json.loads((output / "transforms.json").read_text())
    records = json.loads((output / "projections.json").read_text())
    metrics = json.loads((output / "verification_metrics.json").read_text())
    expected_config, expected_transforms = build_calibration(dataset)
    assert config["dataset"] == "CITIC-MTMC" and config["physical_building_registration"]
    assert config["geometry_estimated_from_video"] is False
    assert len(config["rooms"]) == 4 and len(transforms["cameras"]) == 5
    assert set(transforms["cameras"]) == {"HallWide", "HallSeg", "PF", "PC", "Showroom"}
    shapes = {r["id"]: Polygon(r["polygon"]) for r in config["rooms"]}
    assert all(s.is_valid and s.area > 0 for s in shapes.values())
    union = unary_union(list(shapes.values()))
    assert union.geom_type == "Polygon" and union.is_valid
    assert abs(sum(s.area for s in shapes.values()) - union.area) < 1e-7
    assert len(config["alignment"]["connections"]) == 3
    doorway_errors = []
    for link in config["alignment"]["connections"]:
        doors = []
        for room_key, door_key in [("room_a", "door_a"), ("room_b", "door_b")]:
            room = next(r for r in config["rooms"] if r["id"] == link[room_key])
            door = next(d for d in room["doors"] if d["id"] == link[door_key])
            doors.append(door)
            assert shapes[room["id"]].boundary.distance(Point(door["center"])) < 1e-7
        error = np.linalg.norm(np.array([doors[0]["start"], doors[0]["end"]]) -
                               [doors[1]["end"], doors[1]["start"]], axis=1).max()
        assert error < 1e-7
        doorway_errors.append(float(error))
    for room, expected in zip(config["rooms"], expected_config["rooms"]):
        np.testing.assert_allclose(room["polygon"], expected["polygon"], atol=1e-8)
    max_projection_error = 0.
    accepted, rejected = 0, 0
    methods = {}
    mount_errors, fov_leakage = {}, {}
    for name, entry in transforms["cameras"].items():
        h, hi = np.array(entry["H_image_to_map"]), np.array(entry["H_map_to_image"])
        assert h.shape == (3, 3) and np.isfinite(h).all() and abs(np.linalg.det(h)) > 1e-12
        np.testing.assert_allclose(h @ hi, np.eye(3), atol=1e-8)
        np.testing.assert_allclose(base["cameras"][name]["H_image_to_map"],
                                   expected_transforms["cameras"][name]["H_image_to_map"], atol=1e-9)
        np.testing.assert_allclose(entry["H_image_to_canvas"], np.array(config["T_map_to_canvas"]) @ h, atol=1e-8)
        assert shapes[entry["room_id"]].covers(Point(entry["camera_xy_estimate"]))
        mount = np.asarray(entry["render_mount_xy"])
        mount_error = shapes[entry["room_id"]].boundary.distance(Point(mount))
        assert mount_error < 1e-7
        assert Polygon(entry["render_mount_envelope"]).boundary.distance(Point(mount)) < 1e-7
        assert (np.asarray(entry["camera_xy_estimate"]) - mount) @ entry["render_view_direction"] >= -1e-7
        mount_errors[name] = mount_error
        geometry = shape(entry["render_fov_geometry"])
        assert geometry.is_valid and geometry.area > 0
        leakage = geometry.difference(shapes[entry["room_id"]]).area
        assert leakage < 1e-10
        assert geometry.difference(Polygon(entry["render_mount_envelope"])).area < 1e-10
        for other_name, other_room in shapes.items():
            if other_name != entry["room_id"]:
                assert geometry.intersection(other_room).area < 1e-10
        fov_leakage[name] = leakage
        width, height = config["canvas_size"]
        own_room = next(r for r in config["rooms"] if r["id"] == entry["room_id"])
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.int32(np.rint(project(config["T_map_to_canvas"], own_room["polygon"])))], 255)
        fill, outline = camera_fov_mask(config, entry)
        assert not ((fill > 0) & (mask == 0)).any()
        assert not ((outline > 0) & (mask == 0)).any()
        if entry["extrinsics_available"]:
            r, t, k = np.array(entry["R_world_to_camera"]), np.array(entry["t_world_to_camera"]), np.array(entry["K"])
            np.testing.assert_allclose(-np.linalg.solve(r, t), entry["camera_center_world"], atol=1e-8)
            hp = k @ np.column_stack([r[:, :2], t])
            identity = hp @ h
            np.testing.assert_allclose(identity / identity[2, 2], np.eye(3), atol=1e-8)
        if "overlap_correction" in entry:
            np.testing.assert_allclose(np.array(entry["overlap_correction"]) @ entry["H_image_to_map_before_overlap"], h, atol=1e-9)
    np.testing.assert_allclose(np.mean([e["camera_xy_estimate"] for e in transforms["cameras"].values()], axis=0), [0, 0], atol=1e-8)
    assert len(records) >= 100 and records[-1]["time_seconds"] >= 170.
    from src.citic_evaluation import gt_at_frame
    gt = dataset.ground_truth()
    for record in records:
        truth = gt_at_frame(gt, record["source_frame"], config["T_source_to_map"])
        callouts = camera_label_layout(config, transforms, record, truth)
        rectangles = [v["rect"] for v in callouts.values()]
        assert all(not _overlap(a, b) for i, a in enumerate(rectangles) for b in rectangles[i + 1:])
        people = [p["xy"] for p in record["fused_people"]] + [xy for _, xy in truth]
        for p in project(config["T_map_to_canvas"], people):
            box = (p[0] - 15, p[1] - 28, p[0] + 65, p[1] + 30)
            assert all(not _overlap(r, box) for r in rectangles)
        assert record["time_seconds"] == record["source_frame"] / 25.
        assert len({p["id"] for p in record["fused_people"]}) == len(record["fused_people"])
        by_id = {d["detection_id"]: d for d in record["detections"]}
        for d in record["detections"]:
            # YOLO computes contacts in float32; preserve that precision when replaying it.
            point, method, _ = ground_point(np.asarray(d["keypoints"], np.float32),
                                             np.asarray(d["bbox"], np.float32))
            np.testing.assert_allclose(d["uv"], point, atol=1e-5)
            assert method == d["method"]
            xy = project(transforms["cameras"][d["camera"]]["H_image_to_map"], [d["uv"]])[0]
            error = float(np.linalg.norm(xy - d["xy"]))
            assert error < 1e-8
            max_projection_error = max(error, max_projection_error)
            inside = any(s.covers(Point(xy)) for s in shapes.values())
            assert d["pose_supported"] == pose_supported(d)
            assert d["display_accepted"] == (inside and d["foreground_supported"] and pose_supported(d) and d["floor_supported"])
            assert d["floor_supported"] == (d["floor_support_fraction"] is None or d["floor_support_fraction"] >= 0.5)
            assert d["foreground_supported"] == (d["foreground_fraction"] >= 0.12)
            methods[method] = methods.get(method, 0) + 1
            if d["display_accepted"]:
                accepted += 1
            else:
                rejected += 1
        for person in record["fused_people"]:
            members = [by_id[i] for i in person["member_detection_ids"]]
            assert members and len({d["camera"] for d in members}) == len(members)
            assert all(d["display_accepted"] for d in members)
            positions = np.array([d["xy"] for d in members])
            assert np.linalg.norm(positions[:, None] - positions[None, :], axis=2).max() <= 0.650001
            expected = np.average(positions, axis=0, weights=[d["fusion_weight"] for d in members])
            np.testing.assert_allclose(expected, person["xy"], atol=1e-8)
            assert shapes[person["room_id"]].covers(Point(person["xy"]))
        assert record["headcounts"] == {room: sum(p["room_id"] == room for p in record["fused_people"]) for room in shapes}
    recomputed, _ = evaluate(records, dataset.ground_truth(), config)
    for key in recomputed:
        if isinstance(recomputed[key], float):
            assert abs(recomputed[key] - metrics[key]) < 1e-9
        else:
            assert recomputed[key] == metrics[key]
    assert metrics["doorway_event_comparison"] == evaluate_transitions(
        metrics["inferred_room_transitions"], metrics["ground_truth_room_transitions"])
    # Explicit localization gates, reported alongside full recall/precision and identity switches.
    quality = dict(mean_matched_error_below_075m=metrics["mean_error_m"] <= .75,
                   recall_at_1m_at_least_70pct=metrics["recall_at_1m"] >= .70,
                   precision_at_1m_at_least_70pct=metrics["precision_at_1m"] >= .70)
    assert all(quality.values()), quality
    assert set(metrics["observed_rooms"]) == set(shapes)
    assert all(v > 0 for v in metrics["camera_detection_counts"].values())
    assert len(metrics["inferred_room_transitions"]) > 0
    video = cv2.VideoCapture(str(output / "verification_projection.mp4"))
    video_frames = 0
    while True:
        ok, _ = video.read()
        if not ok:
            break
        video_frames += 1
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    assert video_frames == len(records) and abs(fps - 25. / metrics["sample_stride"]) < 0.01
    sampled_camera_frames = 0
    for c in dataset.cameras:
        for index in [records[0]["source_frame"], config["verification_source_frame"], records[-1]["source_frame"]]:
            frame = dataset.frame(c, index)
            assert frame.shape[:2] == (c["height"], c["width"])
            sampled_camera_frames += 1
    root = Path(__file__).resolve().parents[1]
    for name in ["global_2d_map.png", "verification_bev.png"]:
        assert cv2.imread(str(output / name)) is not None
        assert (output / name).read_bytes() == (root / "assets" / name).read_bytes()
    report = dict(passed=True, dataset="CITIC-MTMC capture1", real_rooms=4, real_cameras=5,
                  physical_building_registration=True, connected_floor_polygon=True,
                  maximum_doorway_alignment_error_m=max(doorway_errors),
                  camera_mount_perimeter_error_m=mount_errors,
                  camera_fov_outside_assigned_room_m2=fov_leakage,
                  collision_free_camera_label_frames=len(records),
                  pose_ground_point_methods=methods, accepted_pose_observations=accepted,
                  rejected_unmodified_observations=rejected, max_projection_error_m=max_projection_error,
                  decoded_source_samples=sampled_camera_frames, decoded_verification_frames=video_frames,
                  localization_quality_gates=quality,
                  mean_matched_error_m=metrics["mean_error_m"], recall_at_1m=metrics["recall_at_1m"],
                  precision_at_1m=metrics["precision_at_1m"], identity_switches_at_2m=metrics["identity_switches_at_2m"],
                  inferred_room_handoffs=len(metrics["inferred_room_transitions"]),
                  doorway_event_precision=metrics["doorway_event_comparison"]["precision"],
                  doorway_event_recall=metrics["doorway_event_comparison"]["recall"],
                  accuracy_scope="Independent interpolated waypoint GT; identities may fragment; published map/calibration, not image-only reconstruction")
    save_json(output / "artifact_validation.json", report)
    save_json(root / "assets/artifact_validation.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/citic_mtmc"))
    args = parser.parse_args()
    validate(args.output_dir, args.data_dir)
