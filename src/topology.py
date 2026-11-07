"""Trajectory doorway discovery, cross-room transition matching and topology configuration."""

from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from .fusion import cosine_distance


def event_appearance(members):
    features = [np.asarray(d.get("transition_appearance", d["appearance"]), float) for d in members]
    mean = np.mean(features, axis=0)
    return (mean / max(np.linalg.norm(mean), 1e-9)).tolist()


def detect_entry_exit(room, records, timing=None):
    timing = timing or {}
    if not records:
        return {"sampled_frames": 0, "uncensored_events": [], "doors": room.get("doors", [])}
    tracks = []
    for record_position, record in enumerate(records):
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
                            "record_position": record_position,
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
                            {
                                "index": record["index"],
                                "record_position": record_position,
                                "xy": c["xy"],
                                "members": c["members"],
                            }
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
            if item["record_position"] < 3 or item["record_position"] > len(records) - 4:
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
                    "time_seconds": records[item["record_position"]]["time_seconds"]
                    + float(timing.get("time_offset_seconds", 0)),
                    "xy": xy.tolist(),
                    "wall_id": wall["id"],
                    "wall_offset_m": offset,
                    "wall_distance_m": float(distance),
                    "source_cameras": [d["camera"] for d in item["members"]],
                    "appearance": event_appearance(item["members"]),
                    "appearance_model": item["members"][0].get(
                        "transition_appearance_model",
                        item["members"][0].get("appearance_model", "legacy-hsv"),
                    ),
                    "clock_id": timing.get("clock_id"),
                    "room_id": room["id"],
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
            "id": f"D{len(doors)+1:02d}",
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
    for door in doors:
        for event in door["events"]:
            event["door_id"] = door["id"]
    return {
        "sampled_frames": len(records),
        "tracklets": len(tracks),
        "uncensored_events": events,
        "mean_shift_clusters": clusters,
        "doors": doors,
        "timing": timing,
        "limitations": "Birth/death tracklets can reflect occlusion; shared-clock, appearance and repeated transition evidence are required for automatic room links.",
    }


def match_transitions(
    rooms,
    dt_min=0.0,
    dt_max=30.0,
    appearance_threshold=0.25,
    min_transitions=2,
    ambiguity_margin=0.04,
):
    if (
        not isinstance(min_transitions, int)
        or min_transitions < 1
        or not 0 < appearance_threshold <= 2
        or ambiguity_margin < 0
    ):
        raise ValueError("Invalid transition support/appearance thresholds")
    if not np.isfinite([dt_min, dt_max]).all() or not 0 <= dt_min < dt_max:
        raise ValueError("Expected 0 <= dt_min < dt_max")
    events = []
    for room in rooms:
        for door in room.get("doors", []):
            for original in door.get("events", []):
                event = dict(original, room_id=room["id"], door_id=door["id"])
                if (
                    event.get("clock_id")
                    and event.get("appearance")
                    and event.get("appearance_model") != "legacy-hsv"
                ):
                    events.append(event)
    exits = [e for e in events if e["kind"] == "exit"]
    entries = [e for e in events if e["kind"] == "entry"]
    if not exits or not entries:
        return [], []
    cost = np.full((len(exits), len(entries)), 1e6)
    for i, source in enumerate(exits):
        for j, dest in enumerate(entries):
            dt = dest["time_seconds"] - source["time_seconds"]
            if source["room_id"] == dest["room_id"] or source["clock_id"] != dest["clock_id"]:
                continue
            if source.get("appearance_model") != dest.get("appearance_model") or len(
                source["appearance"]
            ) != len(dest["appearance"]):
                continue
            if not dt_min < dt < dt_max:
                continue
            distance = cosine_distance(source["appearance"], dest["appearance"])
            if np.isfinite(distance) and distance < appearance_threshold:
                cost[i, j] = distance
    rows, cols = linear_sum_assignment(cost)
    matches = []
    groups = {}
    for i, j in zip(rows, cols):
        value = cost[i, j]
        if value >= 1e6:
            continue
        # Reject ambiguous clothing matches in either direction.
        alternatives = np.r_[np.delete(cost[i], j), np.delete(cost[:, j], i)]
        if len(alternatives) and alternatives.min() - value < ambiguity_margin:
            continue
        a, b = exits[i], entries[j]
        match = {
            "from": f"{a['room_id']}:{a['door_id']}",
            "to": f"{b['room_id']}:{b['door_id']}",
            "dt_seconds": b["time_seconds"] - a["time_seconds"],
            "appearance_distance": float(value),
            "exit_track": [a["room_id"], a["track_id"]],
            "entry_track": [b["room_id"], b["track_id"]],
            "exit_time": a["time_seconds"],
            "entry_time": b["time_seconds"],
            "clock_id": a["clock_id"],
        }
        key = tuple(sorted([match["from"], match["to"]]))
        groups.setdefault(key, []).append(match)
        matches.append(match)
    candidates = []
    for key, support in groups.items():
        independent = {(tuple(m["exit_track"]), tuple(m["entry_track"])) for m in support}
        if len(independent) < min_transitions:
            continue
        candidates.append(
            {
                "from": key[0],
                "to": key[1],
                "gap_m": 0.0,
                "source": "trajectory",
                "supporting_transitions": len(independent),
                "matches": support,
            }
        )
    endpoints = [e for c in candidates for e in [c["from"], c["to"]]]
    # Competing doorway destinations need explicit configuration rather than a guessed topology.
    candidates = [
        c for c in candidates if endpoints.count(c["from"]) == endpoints.count(c["to"]) == 1
    ]
    return candidates, matches


def load_topology(path):
    if path is None:
        return {"links": []}
    path = Path(path)
    result = json.loads(path.read_text())
    if not isinstance(result, dict) or not isinstance(result.get("links", []), list):
        raise ValueError("topology.json must contain a links list")
    return result


def resolve_links(rooms, topology=None):
    topology = topology or {"links": []}
    automatic, matches = match_transitions(rooms, **topology.get("transition", {}))
    links = list(automatic)
    for item in topology.get("links", []):
        link = dict(item, source="config")
        same = [c for c in links if {c["from"], c["to"]} == {link["from"], link["to"]}]
        if same:
            if same[0].get("source") == "trajectory":
                same[0]["gap_m"] = link.get("gap_m", same[0].get("gap_m", 0))
                same[0]["adjacency_also_configured"] = True
            elif float(same[0].get("gap_m", 0)) != float(link.get("gap_m", 0)):
                raise ValueError("Duplicate doorway link has conflicting gap_m values")
            continue
        if any({c["from"], c["to"]} & {link["from"], link["to"]} for c in links):
            raise ValueError("Configured link conflicts with an inferred or configured doorway")
        links.append(link)
    lookup = {r["id"]: r for r in rooms}
    connections = []
    for link in links:
        try:
            ra, da = link["from"].rsplit(":", 1)
            rb, db = link["to"].rsplit(":", 1)
            if ra == rb:
                raise ValueError("A doorway link must connect different rooms")
            for rid, did in [(ra, da), (rb, db)]:
                if rid not in lookup or not any(d["id"] == did for d in lookup[rid]["doors"]):
                    raise ValueError(f"Unknown doorway {rid}:{did}")
            gap = float(link.get("gap_m", 0))
            if not np.isfinite(gap) or gap < 0:
                raise ValueError("gap_m must be finite and nonnegative")
        except (KeyError, TypeError) as error:
            raise ValueError("Each topology link needs from/to room:door endpoints") from error
        connections.append(dict(link, room_a=ra, door_a=da, room_b=rb, door_b=db, gap_m=gap))
    return connections, matches


def declared_doors(room, specifications):
    """Optional measured openings when no pedestrian crossings are recorded."""
    for original in specifications:
        door = deepcopy(original)
        wall = next((w for w in room["walls"] if w["id"] == door.get("wall_id")), None)
        if wall is None:
            index = door.get("wall_index")
            if not isinstance(index, int) or not 0 <= index < len(room["walls"]):
                raise ValueError("Declared door needs a valid wall_id or wall_index")
            wall = room["walls"][index]
        offset, width = float(door["offset_m"]), float(door["width_m"])
        if (
            not np.isfinite([offset, width]).all()
            or offset < 0
            or width <= 0
            or offset + width > wall["length_m"]
        ):
            raise ValueError("Declared door must lie on its wall segment")
        tangent = (np.asarray(wall["end"]) - wall["start"]) / wall["length_m"]
        start = np.asarray(wall["start"]) + tangent * offset
        door.update(
            wall_id=wall["id"],
            start=start.tolist(),
            end=(start + tangent * width).tolist(),
            center=(start + tangent * width / 2).tolist(),
            confirmed=True,
            source="manifest",
            connected_room=None,
        )
        if any(d["id"] == door["id"] for d in room["doors"]):
            raise ValueError("Duplicate doorway ID")
        room["doors"].append(door)
        wall["openings"].append(door["id"])


def analyze_scene(dataset, output, weights=None, device="cpu", frame_count=100):
    """Use longer timelines when provided, otherwise the normal pose acquisition cache."""
    import hashlib
    import cv2
    from .geometry import project, save_json
    from .person_locator import YOLOPoseLocator
    from .appearance import AppearanceEncoder

    output = Path(output)
    config = json.loads((output / "map_config.json").read_text())
    transforms = json.loads((output / "transforms.json").read_text())
    room = config["rooms"][0]
    timelines = []
    path = dataset.root / "timeline_manifest.json"
    if path.exists():
        details = json.loads(path.read_text())
        lookup = {c.get("camera", c.get("id")): c for c in details["cameras"]}
        timelines = [lookup.get(c["id"]) for c in dataset.cameras]
    if not timelines:
        timelines = [
            dict(
                video=c.get("timeline_video"),
                source_start_seconds=c.get("timeline_start_seconds", 0),
            )
            for c in dataset.cameras
        ]
    available = [(c, t) for c, t in zip(dataset.cameras, timelines) if t and t.get("video")]
    timeline_cameras = [c for c, t in available]
    timelines = [t for c, t in available]
    use_timeline = bool(available)
    if use_timeline and len({t.get("source_start_seconds", 0) for t in timelines}) != 1:
        raise ValueError("Align intra-room timeline start times before analyzing trajectories")
    cache = output / "trajectory_detections.json"
    source = output / "trajectory_detections_source.json"
    weights = Path(weights or Path(__file__).resolve().parents[1] / "models/yolo11n-pose.pt")
    if use_timeline:
        paths = [
            Path(t["video"]) if Path(t["video"]).is_absolute() else dataset.root / t["video"]
            for t in timelines
        ]
        if not all(p.exists() for p in paths):
            raise FileNotFoundError("Timeline video does not exist")
        metadata = {
            "weights": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "appearance": "resnet18",
            "appearance_weights": hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "models/resnet18-f37072fd.pth").read_bytes()
            ).hexdigest(),
            "clips": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
        }
        if cache.exists() and source.exists() and json.loads(source.read_text()) == metadata:
            records = json.loads(cache.read_text())
        else:
            locator = YOLOPoseLocator(weights, device)
            encoder = AppearanceEncoder()
            caps = [cv2.VideoCapture(str(p)) for p in paths]
            try:
                rates = [c.get(cv2.CAP_PROP_FPS) for c in caps]
                if min(rates) <= 0 or len(set(rates)) != 1:
                    raise ValueError("Timeline cameras must have equal positive FPS")
                count = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps)
                records = []
                for index in range(count):
                    record = {
                        "index": index,
                        "time_seconds": index / rates[0]
                        + timelines[0].get("source_start_seconds", 0),
                        "detections": [],
                    }
                    for camera, cap in zip(timeline_cameras, caps):
                        ok, frame = cap.read()
                        if not ok:
                            raise RuntimeError("Timeline ended early")
                        detections = [
                            d
                            for d in locator.detect(frame)
                            if d["method"] in ["ankles", "knees_x1.6", "hips_x2"]
                        ]
                        legacy = [d["appearance"] for d in detections]
                        encoder.encode(frame, detections)
                        for d, histogram in zip(detections, legacy):
                            d["transition_appearance"] = d["appearance"]
                            d["transition_appearance_model"] = "resnet18"
                            d["appearance"] = histogram
                            d["camera"] = camera["id"]
                            d.pop("keypoints", None)
                            record["detections"].append(d)
                    records.append(record)
                    if index % 20 == 0:
                        print(room["id"], "entry timeline", index + 1, "/", count, flush=True)
            finally:
                for cap in caps:
                    cap.release()
            save_json(cache, records)
            save_json(source, metadata)
        sample = cv2.VideoCapture(str(paths[0]))
        rate = sample.get(cv2.CAP_PROP_FPS)
        sample.release()
        if rate <= 0:
            raise ValueError("Timeline FPS must be positive")
        for record in records:
            record["time_seconds"] = record["index"] / rate + timelines[0].get(
                "source_start_seconds", 0
            )
    else:
        # Always use the acquisition records, not stale BEV coordinates from a previous build.
        from .observations import collect

        scales = dataset.manifest.get("pose_scales", [1280])
        records = collect(dataset, output, weights, scales, device, frame_count=frame_count)
        records = deepcopy(records)
        for record in records:
            record["time_seconds"] = record["source_frame"] / dataset.cameras[0]["fps"]
            for d in record["detections"]:
                d["transition_appearance_model"] = "resnet18"
    for record in records:
        for d in record["detections"]:
            h = transforms["cameras"][d["camera"]]["H_image_to_map"]
            d["xy"] = project(h, [d["uv"]])[0].tolist()
            u, v = d["uv"]
            box = d["bbox"]
            try:
                d["body_width_m"] = float(
                    np.linalg.norm(np.diff(project(h, [[box[0], v], [box[2], v]]), axis=0))
                )
            except ValueError:
                d["body_width_m"] = 0.0
    declared = [deepcopy(d) for d in room["doors"] if d.get("source") == "manifest"]
    report = detect_entry_exit(room, records, dataset.manifest.get("timing", {}))
    if declared:
        # Measured openings take precedence over inferred body-envelope widths.
        inferred = room["doors"]
        room["doors"] = declared
        for measured in declared:
            nearby = [
                d
                for d in inferred
                if d["wall_id"] == measured["wall_id"]
                and np.linalg.norm(np.asarray(d["center"]) - measured["center"]) < 0.7
            ]
            measured["events"] = [e for d in nearby for e in d["events"]]
        for wall in room["walls"]:
            wall["openings"] = [d["id"] for d in declared if d["wall_id"] == wall["id"]]
    report["doors"] = room["doors"]
    save_json(output / "map_config.json", config)
    save_json(output / "doorway_trajectory_evidence.json", report)
    print(
        room["id"],
        "doorways:",
        len(room["doors"]),
        "shared clock:",
        dataset.manifest.get("timing", {}).get("clock_id"),
        flush=True,
    )
    return report
