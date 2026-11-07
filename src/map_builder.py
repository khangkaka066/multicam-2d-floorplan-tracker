"""Architectural floorplans from structural depth planes and explicit wall geometry."""

from typing import Protocol
from pathlib import Path
import json
import hashlib
import cv2
import numpy as np
from .geometry import project, save_json
from .structure import reconstruct

COLORS = [(210, 105, 40), (40, 145, 220), (160, 75, 165), (90, 145, 65)]


class MapBuilder(Protocol):
    def build(self, dataset, transforms): ...


def configure_canvas(rooms, cameras=None):
    points = np.vstack([np.asarray(room["polygon"]) for room in rooms])
    if cameras:
        points = np.vstack(
            [points, *[np.asarray(c["camera_xy_estimate"]).reshape(1, 2) for c in cameras.values()]]
        )
    lower, upper = points.min(0), points.max(0)
    scale = min(85.0, 1100 / max(upper - lower))
    margin, header = 105, 145
    width = int(np.ceil((upper[0] - lower[0]) * scale)) + 2 * margin
    height = int(np.ceil((upper[1] - lower[1]) * scale)) + header + 110
    matrix = np.array(
        [[scale, 0, margin - lower[0] * scale], [0, -scale, header + upper[1] * scale], [0, 0, 1.0]]
    )
    return {
        "canvas_size": [width, height],
        "T_map_to_canvas": matrix.tolist(),
        "pixels_per_meter": scale,
    }


def draw_floorplan(config, cameras=None):
    width, height = config["canvas_size"]
    matrix = config["T_map_to_canvas"]
    scale = config["pixels_per_meter"]
    image = np.full((height, width, 3), (249, 248, 244), np.uint8)
    ink = (58, 57, 45)
    title = (
        "INDEPENDENT REAL SCENES"
        if config.get("source") == "INDEPENDENT REAL SCENES"
        else "ARCHITECTURAL FLOOR PLAN"
    )
    cv2.putText(image, title, (35, 38), 0, 0.75, ink, 2, cv2.LINE_AA)
    if config.get("alignment", {}).get("status") == "stitched":
        title = "UNIFIED BUILDING FLOOR PLAN"
        image[:50] = (249, 248, 244)
        cv2.putText(image, title, (35, 38), 0, 0.75, ink, 2, cv2.LINE_AA)
    if config.get("alignment", {}).get("status") == "partial":
        image[:50] = (249, 248, 244)
        cv2.putText(image, "PARTIALLY CONNECTED FLOOR PLAN", (35, 38), 0, 0.7, ink, 2, cv2.LINE_AA)
    synthetic = config.get("source", "").startswith("SYNTHETIC")
    subtitle = (
        config.get("subtitle", "Synthetic topology regression fixture")
        if synthetic
        else "Manhattan walls | dimensions estimated from metric depth"
    )
    if config.get("source") == "INDEPENDENT REAL SCENES":
        subtitle = (
            " + ".join(r["id"].upper() for r in config["rooms"])
            + " | placement / shared doorway UNKNOWN"
        )
    cv2.putText(image, subtitle, (35, 65), 0, 0.42, (115, 110, 95), 1, cv2.LINE_AA)
    floor_mask = np.zeros((height, width), np.uint8)
    for i, room in enumerate(config["rooms"]):
        polygon = np.rint(project(matrix, room["polygon"])).astype("int32")
        cv2.fillPoly(image, [polygon], [(237, 231, 216), (225, 237, 229), (238, 226, 235)][i % 3])
        cv2.fillPoly(floor_mask, [polygon], 255)
    for link in config.get("alignment", {}).get("connections", []):
        if link.get("gap_m", 0) <= 0:
            continue
        a = next(r for r in config["rooms"] if r["id"] == link["room_a"])
        b = next(r for r in config["rooms"] if r["id"] == link["room_b"])
        da = next(d for d in a["doors"] if d["id"] == link["door_a"])
        db = next(d for d in b["doors"] if d["id"] == link["door_b"])
        connector = np.int32(project(matrix, [da["start"], da["end"], db["start"], db["end"]]))
        cv2.fillPoly(image, [connector], (237, 231, 216))
        for first, second in [(da["start"], db["end"]), (da["end"], db["start"])]:
            line = np.int32(project(matrix, [first, second]))
            cv2.line(image, tuple(line[0]), tuple(line[1]), ink, 8, cv2.LINE_AA)
    if cameras:
        for i, (name, entry) in enumerate(cameras.items()):
            center = np.asarray(entry["camera_xy_estimate"])
            room = next(r for r in config["rooms"] if r["id"] == entry["room_id"])
            from .boundaries import view_polygon

            direction = np.asarray(entry["optical_axis_world"])
            aim = direction[:2] / np.linalg.norm(direction[:2])
            cone_points = view_polygon(
                center, direction, entry["horizontal_fov_rad"], room["polygon"]
            )
            cone = np.rint(project(matrix, cone_points)).astype("int32")
            mask = np.zeros_like(floor_mask)
            cv2.fillPoly(mask, [cone], 255)
            room_mask = np.zeros_like(floor_mask)
            cv2.fillPoly(room_mask, [np.int32(project(matrix, room["polygon"]))], 255)
            mask = cv2.bitwise_and(mask, room_mask)
            overlay = image.copy()
            overlay[mask > 0] = COLORS[i % 4]
            image = cv2.addWeighted(image, 0.90, overlay, 0.10, 0)
            point = np.rint(project(matrix, [center])[0]).astype(int)
            direction = project(matrix, [center + aim * 0.2])[0] - point
            direction /= max(np.linalg.norm(direction), 1e-8)
            perp = np.array([-direction[1], direction[0]])
            icon = np.array(
                [
                    point + direction * 10,
                    point - direction * 5 + perp * 6,
                    point - direction * 5 - perp * 6,
                ],
                np.int32,
            )
            cv2.fillPoly(image, [icon], COLORS[i % 4])
            cv2.polylines(image, [icon], True, ink, 1, cv2.LINE_AA)
            cv2.putText(
                image,
                name,
                tuple(point + np.array([12, 5])),
                0,
                0.43,
                COLORS[i % 4],
                1,
                cv2.LINE_AA,
            )
    door_labels = {
        (r["id"], d["id"]): d["id"] if d["id"].startswith("D") else f"D{i+1:02d}"
        for r in config["rooms"]
        for i, d in enumerate(r["doors"])
    }
    for room in config["rooms"]:
        doors = {d["id"]: d for d in room["doors"]}
        for wall in room["walls"]:
            a, b = np.asarray(wall["start"]), np.asarray(wall["end"])
            vector = (b - a) / wall["length_m"]
            if wall.get("kind") == "open_scene_limit":
                for distance in np.arange(0, wall["length_m"], 0.25):
                    points = np.int32(
                        project(
                            matrix,
                            [
                                a + vector * distance,
                                a + vector * min(distance + 0.13, wall["length_m"]),
                            ],
                        )
                    )
                    cv2.line(
                        image, tuple(points[0]), tuple(points[1]), (150, 140, 120), 2, cv2.LINE_AA
                    )
                middle = np.int32(project(matrix, [(a + b) / 2])[0])
                cv2.putText(
                    image,
                    "OPEN / extent unknown",
                    tuple(middle + np.array([-70, -12])),
                    0,
                    0.35,
                    (120, 110, 90),
                    1,
                    cv2.LINE_AA,
                )
                continue
            intervals = sorted(
                [
                    (doors[d]["offset_m"], doors[d]["offset_m"] + doors[d]["width_m"])
                    for d in wall["openings"]
                ]
            )
            cursor = 0.0
            for start, end in intervals + [(wall["length_m"], wall["length_m"])]:
                if start > cursor:
                    points = np.rint(
                        project(matrix, [a + vector * cursor, a + vector * start])
                    ).astype(int)
                    cv2.line(image, tuple(points[0]), tuple(points[1]), ink, 8, cv2.LINE_AA)
                cursor = max(cursor, end)
            for door_id in wall["openings"]:
                door = doors[door_id]
                if door.get("connected_room") and room["id"] > door["connected_room"]:
                    continue
                hinge = np.array(door["start"])
                normal = -np.array(wall["outward_normal"])
                closed = np.array(door["end"]) - hinge
                swing = normal * door["width_m"]
                points = np.rint(project(matrix, [hinge, hinge + swing])).astype(int)
                cv2.line(image, tuple(points[0]), tuple(points[1]), ink, 2, cv2.LINE_AA)
                arc = [
                    hinge + closed * np.cos(a) + swing * np.sin(a)
                    for a in np.linspace(0, np.pi / 2, 25)
                ]
                cv2.polylines(
                    image, [np.int32(project(matrix, arc))], False, (135, 128, 110), 1, cv2.LINE_AA
                )
                center = project(matrix, [(hinge + np.array(door["end"])) / 2 + normal * 0.4])[
                    0
                ].astype(int)
                cv2.putText(
                    image,
                    door_labels[(room["id"], door_id)] + ("" if door.get("confirmed") else "*"),
                    tuple(center),
                    0,
                    0.38,
                    (110, 100, 80),
                    1,
                    cv2.LINE_AA,
                )
        from shapely.geometry import Polygon

        p = np.asarray(room["polygon"])
        centroid = np.array(Polygon(p).centroid.coords)[0]
        center = project(matrix, [centroid])[0].astype(int)
        title = room["id"].replace("_", " ").upper()
        size = cv2.getTextSize(title, 0, 0.65, 1)[0]
        cv2.putText(
            image, title, (center[0] - size[0] // 2, center[1] - 10), 0, 0.65, ink, 1, cv2.LINE_AA
        )
        area = room.get("area_m2")
        if area is not None:
            cv2.putText(
                image,
                f"{area:.1f} m2 (est.)",
                (center[0] - 60, center[1] + 15),
                0,
                0.45,
                (115, 110, 95),
                1,
                cv2.LINE_AA,
            )
        if len(p) == 4 and len(config["rooms"]) == 1:
            lo, hi = p.min(0), p.max(0)
            for points, label in [
                ([[lo[0], hi[1] + 0.5], [hi[0], hi[1] + 0.5]], f"{hi[0]-lo[0]:.2f} m"),
                ([[hi[0] + 0.5, lo[1]], [hi[0] + 0.5, hi[1]]], f"{hi[1]-lo[1]:.2f} m"),
            ]:
                a, b = np.rint(project(matrix, points)).astype(int)
                cv2.line(image, tuple(a), tuple(b), (150, 145, 125), 1, cv2.LINE_AA)
                delta = b - a
                normal = np.array([-delta[1], delta[0]], float)
                normal /= np.linalg.norm(normal)
                for point in [a, b]:
                    cv2.line(
                        image,
                        tuple(np.int32(point - normal * 5)),
                        tuple(np.int32(point + normal * 5)),
                        (150, 145, 125),
                        1,
                    )
                middle = (a + b) // 2
                cv2.putText(
                    image,
                    label,
                    tuple(middle + np.array([6, -9])),
                    0,
                    0.44,
                    (110, 105, 90),
                    1,
                    cv2.LINE_AA,
                )
    x, y = 40, height - 58
    cv2.line(image, (x, y), (x + int(scale), y), ink, 3)
    for xx in [x, x + int(scale)]:
        cv2.line(image, (xx, y - 5), (xx, y + 5), ink, 2)
    cv2.putText(image, "1 m (estimated)", (x, y + 22), 0, 0.4, ink, 1, cv2.LINE_AA)
    cv2.putText(
        image,
        (
            "SYNTHETIC geometry | declared doorway adjacency | no real reconstruction data"
            if synthetic
            else "Walls: observed image seams | camera poses: wall-mount prior | dimensions estimated"
        ),
        (35, height - 15),
        0,
        0.36,
        (115, 110, 95),
        1,
        cv2.LINE_AA,
    )
    return image


class ArchitecturalMapBuilder:
    def __init__(self, output):
        self.output = Path(output)

    def build(self, dataset, transforms):
        walls, cloud, report = reconstruct(dataset, transforms, self.output)
        from .boundaries import extract_boundaries, fit_boundary_room, refine_mount_pose

        scale = report["meters_per_epfl_unit_estimate"]
        source_to_metric = np.diag([scale, -scale, 1.0])
        for name, entry in transforms["cameras"].items():
            entry["H_image_to_map"] = (
                source_to_metric @ np.asarray(entry["H_image_to_map"])
            ).tolist()
            entry["H_map_to_image"] = np.linalg.inv(entry["H_image_to_map"]).tolist()
            entry["K_estimated"] = report["cameras"][name]["K"]
        observations = extract_boundaries(dataset, transforms, self.output / "structure")
        room, metric_to_arch, boundary_report = fit_boundary_room(
            observations,
            dataset.manifest["room_id"],
            [report["cameras"][c["id"]]["camera_xy_m"] for c in dataset.cameras],
            allow_open_ends=dataset.manifest.get("allow_open_ends", True),
        )
        source_to_arch = metric_to_arch @ source_to_metric
        transforms["source_units"] = transforms["units"]
        transforms["units"] = "meters"
        transforms["metric_status"] = report["metric_status"]
        transforms["T_epfl_canonical_to_map"] = source_to_arch.tolist()
        transforms["gauge"]["method"] = "center of image-observed Manhattan floor-wall seams"
        mount_report = {}
        for camera in dataset.cameras:
            name = camera["id"]
            entry = transforms["cameras"][name]
            h = metric_to_arch @ np.asarray(entry["H_image_to_map"])
            entry.update(
                {"H_image_to_map": h.tolist(), "H_map_to_image": np.linalg.inv(h).tolist()}
            )
            pose = refine_mount_pose(
                entry,
                camera["width"],
                camera["height"],
                room["polygon"],
                allow_cropped_principal=bool(dataset.manifest.get("plane_rectification_file")),
            )
            mount_report[name] = pose
            entry.update(
                {
                    "camera_xy_estimate": pose["camera_center_world"][:2],
                    "camera_center_world": pose["camera_center_world"],
                    "optical_axis_world": pose["optical_axis_world"],
                    "R_world_to_camera": pose["R_world_to_camera"],
                    "t_world_to_camera": pose["t_world_to_camera"],
                    "camera_position_status": pose["status"],
                    "K_estimated": pose["K"],
                    "horizontal_fov_rad": pose["horizontal_fov_rad"],
                    "wall_clearance_m": pose["wall_clearance_m"],
                }
            )
        from .topology import declared_doors

        declared_doors(room, dataset.manifest.get("doors", []))
        save_json(self.output / "structure/image_boundary_fit.json", boundary_report)
        save_json(self.output / "structure/camera_mount_verification.json", mount_report)
        config = {
            "schema_version": 2,
            "units": "meters",
            "metric_status": report["metric_status"],
            "meters_per_unit": 1.0,
            "estimated_meters_per_epfl_unit": scale,
            "rooms": [room],
            "minimum_display_clearance_m": 0.14,
            "T_epfl_canonical_to_map": source_to_arch.tolist(),
            "structural_source": "structure/reconstruction.json",
            "wall_height_slice_m": [0.1, 2.2],
            "phase3": {
                "status": "single_room",
                "topological_links": [],
                "entry_exit_zones": [],
                "inter_room_alignment": "Core trajectory matching followed by declarative topology fallback",
            },
            "limitations": [
                "Intrinsics and metric scale are estimated, not surveyed",
                "Manhattan closure from image floor-wall seams; camera mounting is a soft pose prior",
                "Doors require uncensored trajectory evidence; no measured inter-scene links",
            ],
        }
        config.update(configure_canvas([room], transforms["cameras"]))
        matrix = np.asarray(config["T_map_to_canvas"])
        for entry in transforms["cameras"].values():
            h = matrix @ np.asarray(entry["H_image_to_map"])
            entry["H_image_to_canvas"] = h.tolist()
            entry["H_canvas_to_image"] = np.linalg.inv(h).tolist()
        image = draw_floorplan(config, transforms["cameras"])
        support = np.full_like(image, 249)
        points = project(matrix, project(metric_to_arch, walls[:, :2]))
        for x, y in np.rint(points[::3]).astype(int):
            if 0 <= x < support.shape[1] and 0 <= y < support.shape[0]:
                cv2.circle(support, (x, y), 1, (170, 145, 90), -1)
        for wall in room["walls"]:
            a, b = np.rint(project(matrix, [wall["start"], wall["end"]])).astype(int)
            cv2.line(support, tuple(a), tuple(b), (45, 45, 45), 2)
        save_json(
            self.output / "structure/manhattan_fit.json",
            {"room": room, "T_metric_to_arch": metric_to_arch.tolist()},
        )
        print(
            "Image-boundary floorplan", room["polygon"], "cameras checked with C=-R.T@t", flush=True
        )
        return image, support, config


def prepare_ground_rectification(dataset, output, force=False):
    """Load a declared rectification or automatically rectify inconsistent ground gauges."""
    from .calibrator import EPFLCalibrator, source_homographies
    from .structure import camera_from_homography, MetricDepthBackend
    from .semantics import SemanticBackend
    from .boundaries import floor_rectification

    declared = dataset.manifest.get("plane_rectification_file")
    if declared and (dataset.root / declared).exists() and not force:
        return EPFLCalibrator().calibrate(dataset)
    if declared and not (dataset.root / declared).exists():
        dataset.manifest.pop("plane_rectification_file")
    provisional = EPFLCalibrator().calibrate(dataset)
    mode = dataset.manifest.get("rectification", "auto")
    incompatible = force or mode == "depth"
    if mode == "auto":
        for camera in dataset.cameras:
            try:
                *_, error = camera_from_homography(
                    provisional["cameras"][camera["id"]], camera["width"], camera["height"]
                )
                if error > dataset.manifest.get("rectification_error_threshold", 0.001):
                    incompatible = True
                    break
            except ValueError:
                incompatible = True
                break
    if not incompatible:
        return provisional
    folder = Path(output) / "structure"
    folder.mkdir(parents=True, exist_ok=True)
    camera = next(
        (c for c in dataset.cameras if c["id"] == dataset.manifest.get("rectification_anchor")),
        dataset.cameras[0],
    )
    image = dataset.frame(camera, 0)
    source = source_homographies(dataset)[camera["id"]]
    if dataset.manifest.get("map_y_axis", "down") == "up":
        source = np.diag([1.0, -1.0, 1.0]) @ source
    # The input gauge, anchor intrinsics and pinned structural-model provenance
    # all affect a rectification, independently of the room name.
    provenance = Path(__file__).resolve().parents[1] / "models/structural_sources.json"
    model_source = provenance.read_bytes() if provenance.exists() else b""
    fingerprint = hashlib.sha256(
        image.tobytes() + source.tobytes() + json.dumps(camera.get("K")).encode() + model_source
    ).hexdigest()
    path = folder / "ground_rectification.json"
    if (
        path.exists()
        and json.loads(path.read_text()).get("input_sha256") == fingerprint
        and not force
    ):
        result = json.loads(path.read_text())
    else:
        depth = MetricDepthBackend(folder).infer(image, camera["id"])
        labels = SemanticBackend(folder).infer(image, camera["id"])
        result = floor_rectification(image, depth, labels, source, camera.get("K"))
        result.update(source_camera=camera["id"], input_sha256=fingerprint)
        save_json(path, result)
    dataset.manifest["plane_rectification_file"] = str(path.resolve())
    return EPFLCalibrator().calibrate(dataset)
