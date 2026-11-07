#!/usr/bin/env python3
"""Refine homographies and emit one tracked, distance-weighted point per person."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from src.geometry import project, save_json
from src.bundle_adjustment import refine_homographies
from src.fusion import fuse_and_track


def process(output):
    output = Path(output)
    records = json.loads((output / "pose_observations.json").read_text())
    transforms = json.loads((output / "transforms.json").read_text())
    config = json.loads((output / "map_config.json").read_text())
    room = config["rooms"][0]
    profile = json.loads((output / "dataset_profile.json").read_text())
    image_sizes = {c["id"]: [c["width"], c["height"]] for c in profile["cameras"]}
    for record in records:
        for d in record["detections"]:
            d["image_size"] = image_sizes[d["camera"]]
            d["lateral_truncation"] = d["bbox"][0] <= 1 or d["bbox"][2] >= d["image_size"][0] - 1
    smoothing_report = {"applied": False, "reason": "original ankle observations retained"}
    for record in records:
        for d in record["detections"]:
            d["xy"] = project(transforms["cameras"][d["camera"]]["H_image_to_map"], [d["uv"]])[
                0
            ].tolist()
    refined, bundle_report = refine_homographies(records, transforms, room)
    canvas = np.asarray(config["T_map_to_canvas"])
    for entry in refined["cameras"].values():
        h = canvas @ np.asarray(entry["H_image_to_map"])
        entry["H_image_to_canvas"] = h.tolist()
        entry["H_canvas_to_image"] = np.linalg.inv(h).tolist()
    for record in records:
        for d in record["detections"]:
            d["xy_before_bundle"] = d["xy"]
            d["xy"] = project(refined["cameras"][d["camera"]]["H_image_to_map"], [d["uv"]])[
                0
            ].tolist()
            clearance = float(
                cv2.pointPolygonTest(np.asarray(room["polygon"], np.float32), tuple(d["xy"]), True)
            )
            d.update(
                inside=clearance > 0,
                boundary_clearance_m=clearance,
                display_accepted=clearance >= config["minimum_display_clearance_m"],
            )
    fusion_report = fuse_and_track(records, refined, room)
    bundle_report["temporal_observation_filter"] = smoothing_report
    for record in records:
        for d in record["detections"]:
            d.pop("keypoints", None)
            d.pop("color_appearance", None)
    save_json(output / "transforms_refined.json", refined)
    save_json(output / "projection_optimization.json", bundle_report)
    save_json(output / "fusion_metrics.json", fusion_report)
    save_json(output / "projections.json", records)
    config["projection_transforms"] = "transforms_refined.json"
    config["render_fused_people"] = True
    config["show_raw_projections"] = False
    config["verification_record_index"] = 78
    save_json(output / "map_config.json", config)
    inspected = min(78, len(records) - 1)
    metrics = {
        "frame_count": len(records),
        "detections": sum(len(r["detections"]) for r in records),
        "source_start_frame": records[0]["source_frame"],
        "inside_room_boundaries": sum(d["inside"] for r in records for d in r["detections"]),
        "displayed_inside_with_clearance": sum(
            d["display_accepted"] for r in records for d in r["detections"]
        ),
        "fused_observations": sum(len(r["fused_people"]) for r in records),
        "fusion": fusion_report,
        "bundle_adjustment": bundle_report,
        "inspected_record_index": inspected,
        "inspected_source_frame": records[inspected]["source_frame"],
        "inspected_headcount": records[inspected]["headcounts"][room["id"]],
        "detection_counts_at_inspected_frame": {
            c: sum(d["camera"] == c for d in records[inspected]["detections"])
            for c in refined["cameras"]
        },
    }
    save_json(output / "verification_metrics.json", metrics)
    print(output, json.dumps(metrics, indent=2), flush=True)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", choices=["laboratory", "passageway"], required=True)
    a = p.parse_args()
    process(Path("outputs") / a.scene)


if __name__ == "__main__":
    main()
