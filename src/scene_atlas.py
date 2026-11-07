#!/usr/bin/env python3
"""Render two real reconstructed scenes without inventing physical adjacency."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import cv2
import numpy as np
from src.geometry import project, save_json
from src.architecture import transform_room
from src.map_builder import configure_canvas, draw_floorplan
from src.bev_visualizer import export_verification
from src.dataset_loader import EPFLDataset


class RealSceneDataset:
    def __init__(self, sources):
        self.cameras = []
        self.sources = {}
        for name, root in sources:
            dataset = EPFLDataset(root)
            for camera in dataset.cameras:
                item = dict(camera, id=f'{name}/{camera["id"]}')
                self.cameras.append(item)
                self.sources[item["id"]] = (dataset, camera)

    def frame(self, camera, index, verification=False):
        dataset, source = self.sources[camera["id"]]
        return dataset.frame(source, index, verification)


def assemble(output, lab, passage, data_root=Path("data")):
    source_configs = {
        name: json.loads((root / "map_config.json").read_text())
        for name, root in [("laboratory", lab), ("passageway", passage)]
    }
    lab_room = source_configs["laboratory"]["rooms"][0]
    passage_room = source_configs["passageway"]["rooms"][0]
    # Pure presentation transform, explicitly NOT a recovered physical building transform.
    shift = (
        np.max(np.asarray(lab_room["polygon"])[:, 0])
        + 3
        - np.min(np.asarray(passage_room["polygon"])[:, 0])
    )
    placement = {
        "laboratory": np.eye(3),
        "passageway": np.array([[1, 0, shift], [0, 1, 0], [0, 0, 1.0]]),
    }
    rooms = [
        transform_room(source_configs[name]["rooms"][0], placement[name])
        for name in ["laboratory", "passageway"]
    ]
    config = {
        "schema_version": 3,
        "units": "meters",
        "coordinate_frame": "unregistered display atlas; metric only within each scene",
        "physical_building_registration": False,
        "rooms": rooms,
        "minimum_display_clearance_m": 0.14,
        "source": "INDEPENDENT REAL SCENES",
        "scene_sources": {
            "laboratory": "../data/manifest.json",
            "passageway": "../data/passageway/manifest.json",
        },
        "alignment": {
            "status": "unknown; separate recordings with no verified shared doorway",
            "connections": [],
            "shared_walls": [],
            "display_gap_m": 3,
            "gap_is_physical_distance": False,
        },
        "scene_transforms": {name: matrix.tolist() for name, matrix in placement.items()},
        "projection_transforms": "transforms_refined.json",
        "recordings_synchronized_across_scenes": False,
        "render_fused_people": True,
        "show_raw_projections": False,
        "verification_record_index": 78,
        "id_scope": "globally unique display IDs; independent recordings, no cross-scene identity claim",
        "limitations": [
            "Scene placement is a page layout, not a building layout",
            "Wall-mount priors refine underconstrained intrinsics",
            "Horizontal FOV follows fitted intrinsics, not forced 60-90 degrees",
            "Passageway has an open unobserved end",
        ],
    }
    config.update(configure_canvas(rooms))
    source_roots = {"laboratory": lab, "passageway": passage}
    combined = {}
    for filename in ["transforms.json", "transforms_refined.json"]:
        combined = {
            "schema_version": 3,
            "units": "meters",
            "coordinate_frame": config["coordinate_frame"],
            "physical_building_registration": False,
            "cameras": {},
        }
        for name, root in source_roots.items():
            original = json.loads((root / filename).read_text())
            matrix = placement[name]
            delta = np.r_[matrix[:2, 2], 0.0]
            prefix = "LAB" if name == "laboratory" else "PASS"
            for camera, entry in original["cameras"].items():
                entry = deepcopy(entry)
                entry["source_scene"] = name
                entry["source_camera"] = camera
                entry["camera_center_scene"] = entry["camera_center_world"]
                entry["camera_center_world"] = (
                    np.asarray(entry["camera_center_world"]) + delta
                ).tolist()
                entry["camera_xy_estimate"] = entry["camera_center_world"][:2]
                entry["t_world_to_camera"] = (
                    np.asarray(entry["t_world_to_camera"])
                    - np.asarray(entry["R_world_to_camera"]) @ delta
                ).tolist()
                if "H_image_to_map_before_bundle" in entry:
                    entry["H_image_to_map_before_bundle"] = (
                        matrix @ np.asarray(entry["H_image_to_map_before_bundle"])
                    ).tolist()
                    entry["footpoint_bundle_correction"] = (
                        matrix
                        @ np.asarray(entry["footpoint_bundle_correction"])
                        @ np.linalg.inv(matrix)
                    ).tolist()
                h = matrix @ np.asarray(entry["H_image_to_map"])
                canvas = np.asarray(config["T_map_to_canvas"]) @ h
                entry.update(
                    H_image_to_map=h.tolist(),
                    H_map_to_image=np.linalg.inv(h).tolist(),
                    H_image_to_canvas=canvas.tolist(),
                    H_canvas_to_image=np.linalg.inv(canvas).tolist(),
                )
                combined["cameras"][f"{prefix}/{camera}"] = entry
        save_json(output / filename, combined)
    image = draw_floorplan(config, combined["cameras"])
    for filename in ["global_2d_map.png", "global_map.png"]:
        cv2.imwrite(str(output / filename), image)
    for name, root in source_roots.items():
        local = json.loads((root / "transforms_refined.json").read_text())
        local_image = draw_floorplan(source_configs[name], local["cameras"])
        cv2.imwrite(str(output / f"room_{name}_map.png"), local_image)
        for filename in ["global_2d_map.png", "global_map.png"]:
            cv2.imwrite(str(root / filename), local_image)
    save_json(output / "map_config.json", config)
    originals = {
        name: json.loads((root / "projections.json").read_text())
        for name, root in source_roots.items()
    }
    id_offsets = {}
    offset = 0
    for name in ["laboratory", "passageway"]:
        id_offsets[name] = offset
        offset += max(
            (p["id"] for r in originals[name] for p in r.get("fused_people", [])), default=0
        )
    records = []
    for index in range(min(map(len, originals.values()))):
        record = {
            "index": index,
            "source_frame": index,
            "frame_note": "Independent recordings; frame pairing is for display only",
            "detections": [],
            "fused_people": [],
            "headcounts": {},
        }
        for name, source_records in originals.items():
            prefix = "LAB" if name == "laboratory" else "PASS"
            for detection in source_records[index]["detections"]:
                d = deepcopy(detection)
                d["source_scene"] = name
                d["source_camera"] = d["camera"]
                d["camera"] = prefix + "/" + d["camera"]
                d["detection_id"] = prefix + ":" + d["detection_id"]
                for key in ["associated_id", "fused_id"]:
                    if key in d:
                        d[key] += id_offsets[name]
                if "xy_before_bundle" in d:
                    d["xy_before_bundle"] = project(placement[name], [d["xy_before_bundle"]])[
                        0
                    ].tolist()
                d["xy"] = project(placement[name], [d["xy"]])[0].tolist()
                record["detections"].append(d)
            for source_person in source_records[index].get("fused_people", []):
                person = deepcopy(source_person)
                person["id"] += id_offsets[name]
                person["xy"] = project(placement[name], [person["xy"]])[0].tolist()
                person["member_cameras"] = [prefix + "/" + c for c in person["member_cameras"]]
                for key in ["member_detection_ids", "candidate_detection_ids"]:
                    person[key] = [prefix + ":" + i for i in person[key]]
                record["fused_people"].append(person)
            record["headcounts"].update(source_records[index].get("headcounts", {}))
        records.append(record)
    save_json(output / "projections.json", records)
    dataset = RealSceneDataset([("LAB", data_root), ("PASS", Path(data_root) / "passageway")])
    export_verification(dataset, records, image, config, output)
    save_json(
        output / "verification_metrics.json",
        {
            "scene_count": 2,
            "camera_count": 8,
            "paired_display_frames": len(records),
            "physical_building_registration": False,
            "detections": sum(len(r["detections"]) for r in records),
            "displayed_inside_with_clearance": sum(
                d["display_accepted"] for r in records for d in r["detections"]
            ),
            "fused_person_observations": sum(len(r["fused_people"]) for r in records),
            "inspected_record_index": min(78, len(records) - 1),
            "inspected_headcounts": records[min(78, len(records) - 1)]["headcounts"],
            "inspected_global_ids": [
                p["id"] for p in records[min(78, len(records) - 1)]["fused_people"]
            ],
            "per_scene": {
                name: json.loads((root / "verification_metrics.json").read_text())
                for name, root in source_roots.items()
            },
        },
    )
    save_json(
        output / "projection_optimization.json",
        {
            "per_scene": {
                name: json.loads((root / "projection_optimization.json").read_text())
                for name, root in source_roots.items()
            }
        },
    )
    print(
        "Rendered two real scenes, eight real cameras. Shared-doorway registration: UNKNOWN.",
        flush=True,
    )


def build_global(output, data_root=Path("data")):
    output = Path(output)
    lab = output / "laboratory"
    passage = output / "passageway"
    if not (lab / "projections.json").exists():
        raise FileNotFoundError("Build and project the Laboratory scene first")
    assemble(output, lab, passage, data_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    build_global(args.output_dir)


if __name__ == "__main__":
    main()
