"""Executable CITIC benchmark using the project's geometry, pose and appearance plugins."""
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .citic_dataset import CiticDataset, build_calibration
from .citic_visualizer import draw_map, verification_panel
from .geometry import project, save_json


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Cannot write {path}")


def collect(dataset, output, transforms, weights, count, stride, imgsz, device):
    from .person_locator import YOLOPoseLocator
    from .appearance import AppearanceEncoder
    from .citic_foreground import TemporalForeground

    cache = output / "citic_pose_cache"
    cache.mkdir(exist_ok=True)
    fingerprint_manifest = dict(dataset.manifest)
    fingerprint_manifest.pop("metadata_matches_archive", None)
    fingerprint = hashlib.sha256(weights.read_bytes() + json.dumps(fingerprint_manifest, sort_keys=True).encode()
                                 + f"{imgsz}:{device}:pose_v1".encode()).hexdigest()
    marker = cache / "fingerprint.txt"
    if marker.exists() and marker.read_text() != fingerprint:
        cache = cache / fingerprint[:12]
        cache.mkdir(exist_ok=True)
        marker = cache / "fingerprint.txt"
    marker.write_text(fingerprint)
    foreground = TemporalForeground(dataset, output)
    detector, encoder = None, None
    records = []
    limit = min(c["frames"] for c in dataset.cameras)
    indices = list(range(0, limit, stride))[:count]
    for i, source_frame in enumerate(indices):
        path = cache / f"{source_frame:06d}.json"
        if path.exists():
            record = json.loads(path.read_text())
        else:
            if detector is None:
                detector = YOLOPoseLocator(weights, device=device, imgsz=imgsz, conf=0.25)
                encoder = AppearanceEncoder()
            record = dict(source_frame=source_frame, time_seconds=source_frame / 25., detections=[])
            for camera in dataset.cameras:
                frame = dataset.frame(camera, source_frame)
                detections = detector.detect(frame)
                encoder.encode(frame, detections)
                foreground.score(frame, camera, detections)
                for j, d in enumerate(detections):
                    d.update(camera=camera["id"], room_id=camera["room_id"],
                             detection_id=f"{source_frame}:{camera['id']}:{j}")
                record["detections"].extend(detections)
            save_json(path, record)
        record["index"] = i
        for camera in dataset.cameras:
            detections = [d for d in record["detections"] if d["camera"] == camera["id"]]
            if detections and any("foreground_fraction" not in d for d in detections):
                foreground.score(dataset.frame(camera, source_frame), camera, detections)
        save_json(path, record)
        for d in record["detections"]:
            d["xy"] = project(transforms["cameras"][d["camera"]]["H_image_to_map"], [d["uv"]])[0].tolist()
        records.append(record)
        if i % 10 == 0 or i == len(indices) - 1:
            print(f"CITIC pose {i + 1}/{len(indices)} / source {source_frame} / {len(record['detections'])} detections", flush=True)
    return records


def run(args, root):
    dataset = CiticDataset(args.data_dir)
    output = args.output_dir or root / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output.resolve() / "yolo_config"))
    import torch
    torch.set_num_threads(2)
    if args.mode == "project":
        if not (output / "transforms.json").exists():
            raise ValueError("Run --mode build-map before project")
        config = json.loads((output / "map_config.json").read_text())
        transforms = json.loads((output / "transforms.json").read_text())
    else:
        config, transforms = build_calibration(dataset)
    if args.mode in ["all", "build-map", "calibrate", "build-global"]:
        save_json(output / "map_config.json", config)
        save_json(output / "transforms.json", transforms)
        save_json(output / "dataset_profile.json", dataset.manifest)
        map_image = draw_map(config, transforms)
        for name in ["global_2d_map.png", "global_map.png"]:
            write_image(output / name, map_image)
        for room in config["rooms"]:
            local_config = dict(config, rooms=[room])
            local_transforms = dict(transforms, cameras={n: e for n, e in transforms["cameras"].items()
                                                        if e["room_id"] == room["id"]})
            write_image(output / f"room_{room['id']}_map.png", draw_map(local_config, local_transforms))
        print("CITIC: four rooms registered through three real shared doorways; five calibrated cameras", flush=True)
    if args.mode not in ["all", "project"]:
        return
    if not (output / "transforms.json").exists():
        raise ValueError("Run --mode build-map before project")
    from .citic_optimization import refine_overlap
    from .building_tracker import BuildingTracker
    from .citic_evaluation import evaluate, gt_at_frame, evaluate_transitions

    records = collect(dataset, output, transforms, args.weights, args.frames, args.sample_stride,
                      args.imgsz or 960, args.device)
    from .citic_foreground import FloorSupport
    floor = FloorSupport(dataset, output)
    camera_by_name = {c["id"]: c for c in dataset.cameras}
    for record in records:
        for d in record["detections"]:
            floor.score(d, camera_by_name[d["camera"]])
    transforms, optimization = refine_overlap(records, transforms)
    for e in transforms["cameras"].values():
        e["H_image_to_canvas"] = (np.asarray(config["T_map_to_canvas"]) @ e["H_image_to_map"]).tolist()
    tracker = BuildingTracker(config, transforms)
    from .citic_foreground import pose_supported
    for record in records:
        for d in record["detections"]:
            xy = project(transforms["cameras"][d["camera"]]["H_image_to_map"], [d["uv"]])[0]
            room = tracker.room_at(xy)
            supported = pose_supported(d)
            d.update(xy=xy.tolist(), pose_supported=supported,
                     display_accepted=room is not None and d["foreground_supported"] and supported and d["floor_supported"],
                     rejection_reason=("outside_building" if room is None else
                                       "static_background" if not d["foreground_supported"] else
                                       "weak_upper_body_pose" if not supported else
                                       "contact_not_on_floor" if not d["floor_supported"] else None),
                     room_id=room or transforms["cameras"][d["camera"]]["room_id"])
        tracker.update(record)
    # GT becomes available only after all inference/association/calibration is complete.
    gt = dataset.ground_truth()
    metrics, matches = evaluate(records, gt, config)
    metrics["doorway_event_comparison"] = evaluate_transitions(tracker.transitions, metrics["ground_truth_room_transitions"])
    metrics.update(sampled_frames=len(records), sample_stride=args.sample_stride,
                   temporal_span_seconds=[records[0]["time_seconds"], records[-1]["time_seconds"]],
                   pose_weights=str(args.weights.name), pose_input_size=args.imgsz or 960,
                   pose_detections=sum(len(r["detections"]) for r in records),
                   rejected_outside_building=sum(d["rejection_reason"] == "outside_building" for r in records for d in r["detections"]),
                   rejected_static_background=sum(d["rejection_reason"] == "static_background" for r in records for d in r["detections"]),
                   rejected_weak_upper_body_pose=sum(d["rejection_reason"] == "weak_upper_body_pose" for r in records for d in r["detections"]),
                   rejected_nonfloor_contacts=sum(d["rejection_reason"] == "contact_not_on_floor" for r in records for d in r["detections"]),
                   floor_support="SegFormer B2 excludes explicit cabinet/mirror/furniture surfaces on 13-frame median backgrounds; ambiguous labels retained; no GT masks",
                   tracker_ids_allocated=tracker.next_id - 1,
                   inferred_room_transitions=tracker.transitions,
                   observed_rooms=sorted({p["room_id"] for r in records for p in r["fused_people"]}),
                   camera_detection_counts={c["id"]: sum(d["camera"] == c["id"] for r in records for d in r["detections"])
                                            for c in dataset.cameras})
    # Choose an interpretable evaluation illustration; all errors remain in whole-sequence metrics.
    matched_by_frame = {}
    for match in matches:
        if match["error_m"] <= 1.:
            matched_by_frame.setdefault(match["source_frame"], []).append(match)
    eligible = [r for r in records if r["fused_people"] and
                len(matched_by_frame.get(r["source_frame"], [])) == len(r["fused_people"]) ==
                len(gt_at_frame(gt, r["source_frame"], config["T_source_to_map"]))]
    selected = max(eligible or records, key=lambda r: (len({d["camera"] for d in r["detections"] if d.get("associated_id")}),
                                          len({p["room_id"] for p in r["fused_people"]}),
                                          sum(d["confidence"] for d in r["detections"])))
    config["verification_selection_method"] = "maximum camera coverage among frames with all predictions/GT matched within 1m; visualization only; all frames remain in metrics"
    config["verification_source_frame"] = selected["source_frame"]
    config["sampled_source_frames"] = [r["source_frame"] for r in records]
    save_json(output / "map_config.json", config)
    save_json(output / "transforms_refined.json", transforms)
    save_json(output / "projection_optimization.json", optimization)
    save_json(output / "projections.json", records)
    save_json(output / "verification_metrics.json", metrics)
    save_json(output / "ground_truth_matches.json", matches)
    save_json(output / "room_transitions.json", dict(inferred=tracker.transitions,
                                                   ground_truth=metrics["ground_truth_room_transitions"],
                                                   note="Inferred IDs are tracker IDs; ground truth IDs are evaluation-only"))
    for name in ["global_2d_map.png", "global_map.png"]:
        write_image(output / name, draw_map(config, transforms))
    truth = gt_at_frame(gt, selected["source_frame"], config["T_source_to_map"])
    write_image(output / "verification_bev.png", verification_panel(dataset, config, transforms, selected, truth, metrics))
    fps = 25. / args.sample_stride
    size = (config["canvas_size"][0], config["canvas_size"][1])
    writer = cv2.VideoWriter(str(output / "verification_projection.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("Cannot open verification video writer")
    try:
        for record in records:
            truth = gt_at_frame(gt, record["source_frame"], config["T_source_to_map"])
            writer.write(draw_map(config, transforms, record, truth))
    finally:
        writer.release()
    assets = root / "assets"
    assets.mkdir(exist_ok=True)
    import shutil
    for name in ["global_2d_map.png", "verification_bev.png", "verification_metrics.json",
                 "map_config.json", "transforms.json", "transforms_refined.json",
                 "projection_optimization.json", "room_transitions.json"]:
        shutil.copyfile(output / name, assets / name)
    print(json.dumps({k: v for k, v in metrics.items() if not k.endswith("transitions")}, indent=2), flush=True)
