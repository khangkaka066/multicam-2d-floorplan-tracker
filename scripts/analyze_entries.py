#!/usr/bin/env python3
"""Locate doorway candidates from uncensored multi-camera trajectory births/deaths."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_loader import EPFLDataset
from src.person_locator import YOLOPoseLocator
from src.geometry import project, save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    output = args.output_dir
    data = EPFLDataset(args.data_dir)
    config = json.loads((output / "map_config.json").read_text())
    transforms = json.loads((output / "transforms.json").read_text())
    room = config["rooms"][0]
    cache = output / "trajectory_detections.json"
    if cache.exists():
        records = json.loads(cache.read_text())
    else:
        locator = YOLOPoseLocator("models/yolo11n-pose.pt", args.device)
        captures = [
            cv2.VideoCapture(str(data.root / c["id"] / "timeline.avi")) for c in data.cameras
        ]
        count = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in captures)
        fps = captures[0].get(cv2.CAP_PROP_FPS)
        records = []
        for index in range(count):
            record = {"index": index, "time_seconds": index / fps, "detections": []}
            for camera, capture in zip(data.cameras, captures):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Timeline ended early")
                h = transforms["cameras"][camera["id"]]["H_image_to_map"]
                for detection in locator.detect(frame):
                    if detection["method"] not in ["ankles", "knees_x1.6", "hips_x2"]:
                        continue
                    point = project(h, [detection["uv"]])[0]
                    if not np.isfinite(point).all():
                        continue
                    uv = detection["uv"]
                    box = detection["bbox"]
                    width = np.linalg.norm(
                        project(h, [[box[0], uv[1]], [box[2], uv[1]]])[1]
                        - project(h, [[box[0], uv[1]], [box[2], uv[1]]])[0]
                    )
                    detection.update(
                        camera=camera["id"], xy=point.tolist(), body_width_m=float(width)
                    )
                    record["detections"].append(detection)
            records.append(record)
            if index % 10 == 0:
                print("Entry timeline", index + 1, "/", count, flush=True)
        for capture in captures:
            capture.release()
        save_json(cache, records)
    # Reproject cached image detections whenever the geometry changes.
    for record in records:
        for d in record["detections"]:
            d["xy"] = project(transforms["cameras"][d["camera"]]["H_image_to_map"], [d["uv"]])[
                0
            ].tolist()
    tracks = []
    for record in records:
        clusters = []
        for d in sorted(record["detections"], key=lambda d: d["confidence"], reverse=True):
            point = np.array(d["xy"])
            appearance = np.array(d["appearance"])
            cluster = next(
                (
                    c
                    for c in clusters
                    if d["camera"] not in c["cameras"]
                    and np.linalg.norm(point - c["xy"]) < 0.50
                    and 1 - c["appearance"] @ appearance < 0.45
                ),
                None,
            )
            if cluster is None:
                clusters.append(
                    {
                        "xy": point,
                        "appearance": appearance,
                        "cameras": [d["camera"]],
                        "members": [d],
                    }
                )
            else:
                cluster["members"].append(d)
                cluster["cameras"].append(d["camera"])
                cluster["xy"] = np.mean([x["xy"] for x in cluster["members"]], 0)
        active = [t for t in tracks if record["index"] - t["last_index"] <= 3]
        used = set()
        if active and clusters:
            distance = np.array(
                [
                    [np.linalg.norm(t["points"][-1]["xy"] - c["xy"]) for c in clusters]
                    for t in active
                ]
            )
            appearance = np.array(
                [[1 - t["appearance"] @ c["appearance"] for c in clusters] for t in active]
            )
            cost = distance + appearance
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if distance[i, j] < 1.2 and appearance[i, j] < 0.6:
                    t = active[i]
                    t["points"].append(
                        {
                            "index": record["index"],
                            "xy": clusters[j]["xy"],
                            "members": clusters[j]["members"],
                        }
                    )
                    t["last_index"] = record["index"]
                    used.add(j)
        for j, c in enumerate(clusters):
            if j not in used:
                tracks.append(
                    {
                        "id": len(tracks),
                        "last_index": record["index"],
                        "appearance": c["appearance"],
                        "points": [
                            {"index": record["index"], "xy": c["xy"], "members": c["members"]}
                        ],
                    }
                )
    events = []
    for track in tracks:
        points = track["points"]
        if len(points) < 6:
            continue
        for kind, item, neighbor in [
            ("entry", points[0], points[min(4, len(points) - 1)]),
            ("exit", points[-1], points[max(0, len(points) - 5)]),
        ]:
            index = item["index"]
            if index < 3 or index > len(records) - 4:
                continue
            xy = item["xy"]
            movement = (neighbor["xy"] - xy) if kind == "entry" else (xy - neighbor["xy"])
            walls = []
            for wall in room["walls"]:
                a = np.array(wall["start"])
                normal = np.array(wall["outward_normal"])
                signed = (xy - a) @ normal
                if abs(signed) > 0.90:
                    continue
                if kind == "entry" and movement @ normal > -0.15:
                    continue
                if kind == "exit" and movement @ normal < 0.15:
                    continue
                walls.append((abs(signed), wall))
            if not walls:
                continue
            distance, wall = min(walls, key=lambda p: p[0])
            a = np.array(wall["start"])
            u = (np.array(wall["end"]) - a) / wall["length_m"]
            offset = float((xy - a) @ u)
            if not 0 <= offset <= wall["length_m"]:
                continue
            events.append(
                {
                    "track_id": track["id"],
                    "kind": kind,
                    "index": index,
                    "time_seconds": records[index]["time_seconds"],
                    "xy": xy.tolist(),
                    "wall_id": wall["id"],
                    "wall_offset_m": offset,
                    "wall_distance_m": float(distance),
                    "source_cameras": [d["camera"] for d in item["members"]],
                    "body_width_m": float(
                        np.median(
                            [
                                d["body_width_m"]
                                for d in item["members"]
                                if 0.2 < d["body_width_m"] < 1.8
                            ]
                        )
                        if any(0.2 < d["body_width_m"] < 1.8 for d in item["members"])
                        else 0
                    ),
                }
            )
    clusters = []
    for event in events:
        center = event["wall_offset_m"]
        same = [e for e in events if e["wall_id"] == event["wall_id"]]
        for _ in range(30):
            neighbors = [e for e in same if abs(e["wall_offset_m"] - center) < 0.55]
            new = float(np.mean([e["wall_offset_m"] for e in neighbors]))
            if abs(new - center) < 1e-4:
                break
            center = new
        if any(
            c["wall_id"] == event["wall_id"] and abs(c["offset_m"] - center) < 0.55
            for c in clusters
        ):
            continue
        members = [e for e in same if abs(e["wall_offset_m"] - center) < 0.55]
        ids = sorted({e["track_id"] for e in members})
        clusters.append(
            {
                "wall_id": event["wall_id"],
                "offset_m": center,
                "independent_tracklets": len(ids),
                "events": members,
            }
        )
    doors = []
    for cluster in sorted(clusters, key=lambda c: c["independent_tracklets"], reverse=True):
        if cluster["independent_tracklets"] < 2:
            continue
        wall = next(w for w in room["walls"] if w["id"] == cluster["wall_id"])
        a = np.array(wall["start"])
        u = (np.array(wall["end"]) - a) / wall["length_m"]
        supported = [e for e in cluster["events"] if e["body_width_m"] > 0]
        if not supported:
            continue
        lo = max(0, min(e["wall_offset_m"] - e["body_width_m"] / 2 for e in supported))
        hi = min(
            wall["length_m"], max(e["wall_offset_m"] + e["body_width_m"] / 2 for e in supported)
        )
        door = {
            "id": f"{room['id']}_door_{len(doors)}",
            "wall_id": wall["id"],
            "offset_m": lo,
            "width_m": hi - lo,
            "start": (a + u * lo).tolist(),
            "end": (a + u * hi).tolist(),
            "center": (a + u * cluster["offset_m"]).tolist(),
            "status": "trajectory-supported entry/exit candidate; width is observed crossing-body envelope, not a jamb measurement",
            "confirmed": False,
            "connected_room": None,
            "supporting_tracklets": cluster["independent_tracklets"],
            "events": cluster["events"],
        }
        doors.append(door)
    for w in room["walls"]:
        w["openings"] = []
    for door in doors:
        next(w for w in room["walls"] if w["id"] == door["wall_id"])["openings"].append(door["id"])
    room["doors"] = doors
    save_json(output / "map_config.json", config)
    save_json(
        output / "doorway_trajectory_evidence.json",
        {
            "sampled_frames": len(records),
            "sampling_fps": 2,
            "duration_seconds": records[-1]["time_seconds"],
            "tracklets": len(tracks),
            "uncensored_events": events,
            "mean_shift_clusters": clusters,
            "doors": doors,
            "limitations": "Tracklets may fragment or be born after occlusion; candidate locations require physical confirmation.",
        },
    )
    print(
        json.dumps(
            {
                "events": len(events),
                "clusters": len(clusters),
                "doors": [(d["wall_id"], d["center"], d["supporting_tracklets"]) for d in doors],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
