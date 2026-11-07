"""Real building geometry, camera visibility and synchronized detector/GT evidence."""
import cv2
import numpy as np
from shapely.geometry import shape

from .geometry import project

COLORS = [(65, 115, 220), (185, 95, 65), (90, 155, 80), (165, 90, 160), (80, 160, 185)]
ID_COLORS = [(60, 100, 225), (195, 140, 25), (120, 170, 60), (170, 80, 170)]
ROOM_LABEL_POSITIONS = {"passageway_pf": [0.9, 2.6], "passageway_pc": [14.8, 6.3],
                        "showroom": [8.7, 11.5], "entrance_hall": [26.8, 10.5]}
ROOM_LABELS = {"passageway_pf": "PF", "passageway_pc": "PASSAGEWAY PC",
               "showroom": "SHOWROOM", "entrance_hall": "ENTRANCE HALL"}


def text(image, label, xy, scale=0.48, color=(65, 61, 53), thickness=1):
    cv2.putText(image, label, tuple(np.int32(xy)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def camera_fov_mask(config, camera):
    """Both fill and antialiased outlines are restricted to the assigned room pixels."""
    width, height = config["canvas_size"]
    matrix = config["T_map_to_canvas"]
    room = next(r for r in config["rooms"] if r["id"] == camera["room_id"])
    room_mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(room_mask, [np.int32(np.rint(project(matrix, room["polygon"])))], 255)
    geometry = shape(camera["render_fov_geometry"])
    mask, outline = np.zeros_like(room_mask), np.zeros_like(room_mask)
    for polygon in getattr(geometry, "geoms", [geometry]):
        points = np.int32(np.rint(project(matrix, np.asarray(polygon.exterior.coords))))
        cv2.fillPoly(mask, [points], 255)
        cv2.polylines(outline, [points], True, 255, 1, cv2.LINE_AA)
        for ring in polygon.interiors:
            points = np.int32(np.rint(project(matrix, np.asarray(ring.coords))))
            cv2.fillPoly(mask, [points], 0)
            cv2.polylines(outline, [points], True, 255, 1, cv2.LINE_AA)
    return cv2.bitwise_and(mask, room_mask), cv2.bitwise_and(outline, room_mask)


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def camera_label_layout(config, transforms, record=None, gt=None):
    """Prefer short exterior callouts; reserve doors, people, room titles and other labels."""
    width, height = config["canvas_size"]
    matrix = np.asarray(config["T_map_to_canvas"])
    floor = np.zeros((height, width), np.uint8)
    walls = np.zeros_like(floor)
    occupied = []
    for room in config["rooms"]:
        cv2.fillPoly(floor, [np.int32(project(matrix, room["polygon"]))], 1)
        cv2.polylines(walls, [np.int32(np.rint(project(matrix, room["polygon"])))], True, 1, 8)
        p = project(matrix, project(config["T_source_to_map"], [ROOM_LABEL_POSITIONS[room["id"]]]))[0]
        (w, h), base = cv2.getTextSize(ROOM_LABELS[room["id"]], 0, .47, 1)
        occupied.append((p[0] - w / 2 - 4, p[1] - h - 4, p[0] + w / 2 + 4, p[1] + base + 4))
        for door in room["doors"]:
            p = project(matrix, [door["center"]])[0]
            occupied.append((p[0] - 8, p[1] - 25, p[0] + 45, p[1] + 9))
    people = [p["xy"] for p in record["fused_people"]] if record else []
    people += [xy for _, xy in gt] if gt is not None else []
    for p in project(matrix, people):
        occupied.append((p[0] - 15, p[1] - 28, p[0] + 65, p[1] + 30))
    anchors = {n: project(matrix, [e["render_mount_xy"]])[0] for n, e in transforms["cameras"].items()}
    occupied.extend((p[0] - 14, p[1] - 14, p[0] + 14, p[1] + 14) for p in anchors.values())
    integral = cv2.integral(floor)
    wall_integral = cv2.integral(walls)
    occupied.append((width - 240, height - 100, width - 20, height - 60))  # scale bar
    layouts = {}
    for name, entry in transforms["cameras"].items():
        anchor = anchors[name]
        direction = matrix[:2, :2] @ entry["render_view_direction"]
        direction /= np.linalg.norm(direction)
        normal = -direction
        tangent = np.array([-normal[1], normal[0]])
        (w, h), base = cv2.getTextSize(name, 0, .44, 1)
        w, h = w + 12, h + base + 10
        candidates = [anchor + normal * radius + tangent * offset
                      for radius in [30, 50, 75, 105, 140]
                      for offset in [0, 30, -30, 65, -65, 110, -110]]
        candidates += [anchor + np.array([dx, dy])
                       for dx in [-80, -50, 0, 50, 80] for dy in [-90, -45, -25, 0, 25, 45, 90]]
        valid = []
        for p in candidates:
            x, y = np.rint(p - [w / 2, h / 2]).astype(int)
            rect = (x, y, x + w, y + h)
            if x < 12 or y < 108 or x + w > width - 12 or y + h > height - 80:
                continue
            if any(_overlap(rect, other) for other in occupied):
                continue
            wall_pixels = wall_integral[y + h, x + w] - wall_integral[y, x + w] - wall_integral[y + h, x] + wall_integral[y, x]
            if wall_pixels:
                continue
            covered = integral[y + h, x + w] - integral[y, x + w] - integral[y + h, x] + integral[y, x]
            # Keep names off the active floor whenever a nearby exterior position is available.
            score = np.linalg.norm(p - anchor) + 100 * covered / (w * h)
            valid.append((score, rect))
        if not valid:
            raise ValueError(f"No collision-free camera label position for {name}")
        _, rect = min(valid, key=lambda item: item[0])
        occupied.append(rect)
        x, y, _, _ = rect
        layouts[name] = dict(rect=rect, baseline=(x + 6, y + h - base - 5),
                             anchor=anchor.tolist(),
                             leader_end=np.clip(anchor, [rect[0], rect[1]], [rect[2], rect[3]]).tolist())
    return layouts


def draw_map(config, transforms, record=None, gt=None):
    width, height = config["canvas_size"]
    # CITIC source Y points down; preserve the familiar orientation of the published plan.
    matrix = np.asarray(config["T_map_to_canvas"])
    image = np.full((height, width, 3), (249, 248, 244), np.uint8)
    text(image, "CITIC / ONE CONNECTED BUILDING", (32, 37), 0.78, thickness=2)
    text(image, f"{len(config['rooms'])} monitored spaces / {len(transforms['cameras'])} real cameras / capture1 / metric shared-door registration", (32, 64), 0.46)
    text(image, "Published floor geometry + calibration | HA3 vestibule retained | door widths nominal", (32, 87), 0.42)
    for i, room in enumerate(config["rooms"]):
        poly = np.int32(np.rint(project(matrix, room["polygon"])))
        cv2.fillPoly(image, [poly], [(232, 235, 220), (229, 238, 237),
                                     (235, 228, 239), (237, 232, 219)][i])
    for i, (name, camera) in enumerate(transforms["cameras"].items()):
        mask, outline = camera_fov_mask(config, camera)
        overlay = image.copy()
        overlay[mask > 0] = COLORS[i]
        image = cv2.addWeighted(image, 0.85, overlay, 0.15, 0)
        image[outline > 0] = COLORS[i]
    for room in config["rooms"]:
        for wall in room["walls"]:
            start, end = np.array(wall["start"]), np.array(wall["end"])
            tangent = (end - start) / wall["length_m"]
            openings = sorted((d["offset_m"], d["offset_m"] + d["width_m"])
                              for d in room["doors"] if d["wall_id"] == wall["id"])
            cursor = 0.
            for lo, hi in openings + [(wall["length_m"], wall["length_m"])]:
                pts = np.int32(project(matrix, [start + cursor * tangent, start + lo * tangent]))
                cv2.line(image, tuple(pts[0]), tuple(pts[1]), (78, 71, 61), 4, cv2.LINE_AA)
                cursor = hi
        for door in room["doors"]:
            if room["id"] > door["connected_room"]:
                continue
            pts = np.int32(project(matrix, [door["start"], door["end"]]))
            cv2.line(image, tuple(pts[0]), tuple(pts[1]), (130, 160, 175), 3, cv2.LINE_AA)
            center = np.int32(project(matrix, [door["center"]])[0])
            text(image, door["id"], center + [5, -8], 0.36)
    for room in config["rooms"]:
        xy = project(matrix, project(config["T_source_to_map"], [ROOM_LABEL_POSITIONS[room["id"]]]))[0]
        label = ROOM_LABELS[room["id"]]
        size = cv2.getTextSize(label, 0, 0.47, 1)[0]
        text(image, label, xy - [size[0] / 2, 0], 0.47)
    callouts = camera_label_layout(config, transforms, record, gt)
    for i, (name, camera) in enumerate(transforms["cameras"].items()):
        center = project(matrix, [camera["render_mount_xy"]])[0]
        axis = np.asarray(matrix)[:2, :2] @ camera["render_view_direction"]
        axis /= np.linalg.norm(axis)
        tangent = np.array([-axis[1], axis[0]])
        triangle = np.int32(np.rint([center + axis * 12, center - axis * 4 + tangent * 6,
                                     center - axis * 4 - tangent * 6]))
        label = callouts[name]
        cv2.line(image, tuple(np.int32(center)), tuple(np.int32(label["leader_end"])), (140, 137, 128), 1, cv2.LINE_AA)
        cv2.rectangle(image, label["rect"][:2], label["rect"][2:], (249, 248, 244), -1)
        text(image, name, label["baseline"], .44, COLORS[i])
        cv2.fillPoly(image, [triangle], COLORS[i], cv2.LINE_AA)
        cv2.polylines(image, [triangle], True, (65, 60, 50), 1, cv2.LINE_AA)
    if gt is not None:
        for identity, xy in gt:
            point = np.int32(project(matrix, [xy])[0])
            cv2.drawMarker(image, tuple(point), (35, 35, 35), cv2.MARKER_CROSS, 15, 2)
            text(image, f"GT {identity}", point + [8, 17], 0.37)
    if record:
        for person in record["fused_people"]:
            point = np.int32(project(matrix, [person["xy"]])[0])
            color = ID_COLORS[(person["id"] - 1) % len(ID_COLORS)]
            cv2.circle(image, tuple(point), 8, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, tuple(point), 6, color, -1, cv2.LINE_AA)
            text(image, f"ID {person['id']}", point + [9, -8], 0.43, color, 2)
        text(image, f"t={record['time_seconds']:.2f}s | frame {record['source_frame']} | colored: predicted IDs / black crosses: waypoint GT",
             (32, height - 54), 0.42)
    else:
        text(image, "FOV: first wall / assigned room | camera triangles: inferred perimeter mounts from published poses", (32, height - 54), 0.42)
    text(image, "CITIC-MTMC / Carro-Lagoa et al. / CC BY-NC-SA 4.0 / doi:10.5281/zenodo.17511785", (32, height - 25), 0.4)
    scale = config["pixels_per_meter"]
    cv2.line(image, (width - 230, height - 70), (width - 230 + round(5 * scale), height - 70), (70, 65, 60), 3)
    text(image, "5 meters", (width - 225, height - 82), 0.4)
    return image


def verification_panel(dataset, config, transforms, record, gt, metrics):
    map_image = draw_map(config, transforms, record, gt)
    panel_width, tile_width, tile_height = 1680, 560, 360
    map_height = round(map_image.shape[0] * panel_width / map_image.shape[1])
    image = np.full((map_height + tile_height * 2 + 80, panel_width, 3), 246, np.uint8)
    image[:map_height] = cv2.resize(map_image, (panel_width, map_height))
    for i, camera in enumerate(dataset.cameras):
        frame = dataset.frame(camera, record["source_frame"])
        for detection in record["detections"]:
            if detection["camera"] != camera["id"]:
                continue
            identity = detection.get("associated_id")
            color = ID_COLORS[(identity - 1) % len(ID_COLORS)] if identity else (130, 130, 130)
            x1, y1, x2, y2 = np.int32(detection["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            point = tuple(np.int32(detection["uv"]))
            cv2.circle(frame, point, 6, color, -1)
            label = f"ID {identity}" if identity else "outside / rejected"
            text(frame, label + " / " + detection["method"], [x1, max(22, y1 - 8)], 0.6, color, 2)
        x, y = (i % 3) * tile_width, map_height + (i // 3) * tile_height
        scale = min((tile_width - 16) / frame.shape[1], (tile_height - 50) / frame.shape[0])
        resized = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
        text(image, camera["id"] + " / " + camera["room_id"].replace("_", " "), (x + 12, y + 26), 0.52, thickness=2)
        xx = x + (tile_width - resized.shape[1]) // 2
        image[y + 40:y + 40 + resized.shape[0], xx:xx + resized.shape[1]] = resized
    x, y = 1140, map_height + tile_height + 32
    error_label = f"{metrics['mean_error_m']:.2f} m" if metrics['mean_error_m'] is not None else "n/a"
    for line in ["WAYPOINT GT COMPARISON", "GT is never an inference input",
                 f"Matched: {metrics['matched_positions']} / GT: {metrics['ground_truth_positions']}",
                 f"Mean error: {error_label}",
                 f"Recall @ 1m: {metrics['recall_at_1m']:.1%}",
                 f"Precision @ 1m: {metrics['precision_at_1m']:.1%}",
                 "GT: interpolated phone waypoints",
                 "Colored IDs are tracker outputs",
                 "Single-view observations retained"]:
        text(image, line, (x, y), 0.48)
        y += 28
    text(image, "Real synchronized capture1 frames. No person positions are snapped to ground truth or room boundaries.",
         (24, image.shape[0] - 30), 0.56)
    return image
