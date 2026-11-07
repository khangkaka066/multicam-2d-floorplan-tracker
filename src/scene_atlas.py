"""Topology-aware building assembly with an explicit disconnected-scene atlas fallback."""

from copy import deepcopy
import json
import os
from pathlib import Path
import cv2
import numpy as np
from .geometry import project, save_json
from .architecture import layout_rooms, transform_room
from .map_builder import configure_canvas, draw_floorplan
from .bev_visualizer import export_verification
from .dataset_loader import RoomDataset, discover_rooms
from .topology import load_topology


class RealSceneDataset:
    def __init__(self, sources):
        self.cameras = []
        self.sources = {}
        for name, source in sources:
            dataset = source if isinstance(source, RoomDataset) else RoomDataset(source)
            for camera in dataset.cameras:
                item = dict(
                    camera, id=f'{name}/{camera["id"]}', source_scene=dataset.manifest["room_id"]
                )
                self.cameras.append(item)
                self.sources[item["id"]] = (dataset, camera)

    def frame(self, camera, index, verification=False):
        dataset, source = self.sources[camera["id"]]
        return dataset.frame(source, index, verification)


def transform_camera(entry, matrix, canvas):
    """Apply the same SE(2) gauge to H, C, optical axis, R and t (including rotations)."""
    entry = deepcopy(entry)
    matrix = np.asarray(matrix)
    q = np.eye(3)
    q[:2, :2] = matrix[:2, :2]
    delta = np.r_[matrix[:2, 2], 0.0]
    if "camera_center_world" in entry:
        entry["camera_center_scene"] = entry["camera_center_world"]
        entry["camera_center_world"] = (q @ entry["camera_center_world"] + delta).tolist()
        entry["camera_xy_estimate"] = entry["camera_center_world"][:2]
        entry["optical_axis_world"] = (q @ entry["optical_axis_world"]).tolist()
        rotation = np.asarray(entry["R_world_to_camera"]) @ q.T
        entry["R_world_to_camera"] = rotation.tolist()
        entry["t_world_to_camera"] = (
            np.asarray(entry["t_world_to_camera"]) - rotation @ delta
        ).tolist()
    elif "camera_xy_estimate" in entry:
        entry["camera_xy_estimate"] = project(matrix, [entry["camera_xy_estimate"]])[0].tolist()
    if "H_image_to_map_before_bundle" in entry:
        entry["H_image_to_map_before_bundle"] = (
            matrix @ entry["H_image_to_map_before_bundle"]
        ).tolist()
        entry["footpoint_bundle_correction"] = (
            matrix @ entry["footpoint_bundle_correction"] @ np.linalg.inv(matrix)
        ).tolist()
    h = matrix @ entry["H_image_to_map"]
    pixel = np.asarray(canvas) @ h
    entry.update(
        H_image_to_map=h.tolist(),
        H_map_to_image=np.linalg.inv(h).tolist(),
        H_image_to_canvas=pixel.tolist(),
        H_canvas_to_image=np.linalg.inv(pixel).tolist(),
    )
    return entry


def display_schedule(originals, datasets):
    """Align shared-clock clips by source time; independent clips pair indices only for display."""
    timings = {name: d.manifest.get("timing", {}) for name, d in datasets.items()}
    clocks = [t.get("clock_id") for t in timings.values()]
    common = len(clocks) > 1 and bool(clocks[0]) and len(set(clocks)) == 1
    if not common:
        return [
            {name: i for name in originals} for i in range(min(map(len, originals.values())))
        ], False
    stamps = {}
    for name, records in originals.items():
        fps = datasets[name].cameras[0]["fps"]
        stamps[name] = np.array(
            [
                r.get("time_seconds", r["source_frame"] / fps)
                + timings[name].get("time_offset_seconds", 0)
                for r in records
            ]
        )
    times = sorted({round(float(t), 6) for values in stamps.values() for t in values})
    schedule = []
    for timestamp in times:
        indices = {}
        for name, values in stamps.items():
            nearest = int(np.argmin(abs(values - timestamp)))
            indices[name] = (
                nearest
                if abs(values[nearest] - timestamp) <= 0.51 / datasets[name].cameras[0]["fps"]
                else None
            )
        schedule.append(indices)
    return schedule, True


def assemble(output, datasets, layout="stitched", topology_path=None, maps_only=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    datasets = {d.manifest["room_id"]: d for d in datasets}
    roots = {name: output / name for name in datasets}
    configs = {
        name: json.loads((root / "map_config.json").read_text()) for name, root in roots.items()
    }
    local_rooms = [configs[name]["rooms"][0] for name in datasets]
    rooms, alignment = layout_rooms(local_rooms, load_topology(topology_path), layout)
    placement = {room["id"]: np.asarray(room["T_local_to_global"]) for room in rooms}
    registered = alignment["physical_building_registration"]
    if registered:
        centers = []
        for name, root in roots.items():
            local = json.loads((root / "transforms.json").read_text())
            centers.extend(
                project(
                    placement[name], [e["camera_xy_estimate"] for e in local["cameras"].values()]
                )
            )
        gauge = np.eye(3)
        gauge[:2, 2] = -np.mean(centers, axis=0)
        rooms = [transform_room(room, gauge) for room in rooms]
        placement = {name: gauge @ matrix for name, matrix in placement.items()}
        alignment["gauge"] = {
            "method": "camera-network centroid",
            "T_before_to_canonical": gauge.tolist(),
        }
    config = {
        "schema_version": 3,
        "units": "meters",
        "coordinate_frame": (
            "topology-registered building; estimated local metric geometry"
            if registered
            else "unregistered display atlas; metric only within each scene"
        ),
        "physical_building_registration": registered,
        "rooms": rooms,
        "minimum_display_clearance_m": min(
            c.get("minimum_display_clearance_m", 0.14) for c in configs.values()
        ),
        "source": (
            "STITCHED BUILDING"
            if registered
            else (
                "PARTIAL BUILDING"
                if alignment["status"] == "partial"
                else "INDEPENDENT REAL SCENES"
            )
        ),
        "scene_sources": {
            name: os.path.relpath(d.root / "manifest.json", output) for name, d in datasets.items()
        },
        "alignment": alignment,
        "scene_transforms": {n: m.tolist() for n, m in placement.items()},
        "projection_transforms": "transforms_refined.json",
        "render_fused_people": True,
        "show_raw_projections": False,
        "verification_record_index": 78,
        "id_scope": "globally unique display IDs; cross-room identity continuity is not asserted",
        "limitations": [
            "Dimensions and camera intrinsics are estimated, not surveyed",
            "Only connected topology components have an evidenced relative pose",
        ],
    }
    config.update(configure_canvas(rooms))
    prefixes = {name: d.manifest.get("camera_prefix", name.upper()) for name, d in datasets.items()}
    if len(set(prefixes.values())) != len(prefixes):
        raise ValueError("Camera prefixes must be unique across rooms")
    combined = None
    for filename in ["transforms.json", "transforms_refined.json"]:
        combined = {
            "schema_version": 3,
            "units": "meters",
            "coordinate_frame": config["coordinate_frame"],
            "physical_building_registration": registered,
            "cameras": {},
        }
        for name, root in roots.items():
            source = root / filename
            if maps_only or not source.exists():
                source = root / "transforms.json"
            original = json.loads(source.read_text())
            for camera, entry in original["cameras"].items():
                transformed = transform_camera(entry, placement[name], config["T_map_to_canvas"])
                transformed.update(source_scene=name, source_camera=camera)
                combined["cameras"][f"{prefixes[name]}/{camera}"] = transformed
        save_json(output / filename, combined)
    image = draw_floorplan(config, combined["cameras"])
    for filename in ["global_2d_map.png", "global_map.png"]:
        cv2.imwrite(str(output / filename), image)
    save_json(output / "map_config.json", config)
    save_json(output / "topology.json", alignment)
    if maps_only:
        return config
    originals = {
        name: json.loads((root / "projections.json").read_text()) for name, root in roots.items()
    }
    if any(not records for records in originals.values()):
        raise ValueError("Cannot assemble empty person projections")
    schedule, synchronized = display_schedule(originals, datasets)
    config["recordings_synchronized_across_scenes"] = synchronized
    config["verification_record_index"] = min(78, len(schedule) - 1)
    offsets = {}
    offset = 0
    for name in datasets:
        offsets[name] = offset
        offset += max(
            (p["id"] for r in originals[name] for p in r.get("fused_people", [])), default=0
        )
    records = []
    for index, indices in enumerate(schedule):
        record = {
            "index": index,
            "source_frame": index,
            "frame_note": (
                "Shared-clock streams aligned by source timestamps"
                if synchronized
                else "Independent recordings; frame pairing is for display only"
            ),
            "detections": [],
            "fused_people": [],
            "headcounts": {name: 0 for name in datasets},
            "camera_frame_indices": {
                f"{prefixes[name]}/{c['id']}": indices[name]
                for name, d in datasets.items()
                for c in d.cameras
            },
        }
        for name, local_index in indices.items():
            if local_index is None:
                continue
            source_record = originals[name][local_index]
            prefix = prefixes[name]
            for original in source_record["detections"]:
                d = deepcopy(original)
                d.update(source_scene=name, source_camera=d["camera"])
                d["camera"] = prefix + "/" + d["camera"]
                d["detection_id"] = prefix + ":" + d["detection_id"]
                for key in ["associated_id", "fused_id"]:
                    if key in d:
                        d[key] += offsets[name]
                for key in ["xy", "xy_before_bundle"]:
                    if key in d:
                        d[key] = project(placement[name], [d[key]])[0].tolist()
                record["detections"].append(d)
            for original in source_record.get("fused_people", []):
                person = deepcopy(original)
                person["id"] += offsets[name]
                person["xy"] = project(placement[name], [person["xy"]])[0].tolist()
                person["member_cameras"] = [prefix + "/" + c for c in person["member_cameras"]]
                for key in ["member_detection_ids", "candidate_detection_ids"]:
                    person[key] = [prefix + ":" + i for i in person[key]]
                record["fused_people"].append(person)
            record["headcounts"].update(source_record.get("headcounts", {}))
        records.append(record)
    save_json(output / "projections.json", records)
    save_json(output / "map_config.json", config)
    source_data = RealSceneDataset([(prefixes[name], d) for name, d in datasets.items()])
    export_verification(source_data, records, image, config, output)
    selected = records[config["verification_record_index"]]
    metrics = {
        "scene_count": len(datasets),
        "camera_count": len(combined["cameras"]),
        "paired_display_frames": len(records),
        "physical_building_registration": registered,
        "layout_status": alignment["status"],
        "detections": sum(len(r["detections"]) for r in records),
        "displayed_inside_with_clearance": sum(
            d["display_accepted"] for r in records for d in r["detections"]
        ),
        "fused_person_observations": sum(len(r["fused_people"]) for r in records),
        "inspected_record_index": selected["index"],
        "inspected_headcounts": selected["headcounts"],
        "inspected_global_ids": [p["id"] for p in selected["fused_people"]],
        "per_scene": {
            name: json.loads((root / "verification_metrics.json").read_text())
            for name, root in roots.items()
        },
    }
    save_json(output / "verification_metrics.json", metrics)
    save_json(
        output / "projection_optimization.json",
        {
            "per_scene": {
                name: json.loads((root / "projection_optimization.json").read_text())
                for name, root in roots.items()
            }
        },
    )
    print(
        f"Rendered {len(datasets)} manifest rooms, {len(combined['cameras'])} cameras; layout={alignment['status']}; links={len(alignment['connections'])}",
        flush=True,
    )
    return config


def build_global(
    output, data_root=Path("data"), layout="stitched", topology_path=None, maps_only=False
):
    datasets = discover_rooms(data_root)
    return assemble(output, datasets, layout, topology_path, maps_only)
