"""Image-observed floor/wall seams, robust Manhattan lines and pose consistency."""

import cv2
import numpy as np
from scipy.optimize import least_squares
from .geometry import project, save_json
from .architecture import room_from_polygon


def extract_boundaries(dataset, transforms, folder):
    from pathlib import Path

    folder = Path(folder)
    observations = []
    for camera in dataset.cameras:
        name = camera["id"]
        labels = np.load(folder / f"{name}_semantics.npy")
        floor = np.isin(labels, [3, 28]).astype("uint8")
        wall = (labels == 0).astype("uint8")
        # Keep actual floor pixels touching wall labels, not silhouettes of furniture.
        seam = (floor > 0) & (cv2.dilate(wall, np.ones((5, 5), np.uint8)) > 0)
        seam[: int(labels.shape[0] * 0.18)] = False
        ys, xs = np.where(seam)
        uv = np.c_[xs, ys]
        xy = project(transforms["cameras"][name]["H_image_to_map"], uv)
        valid = np.isfinite(xy).all(1) & (np.linalg.norm(xy, axis=1) < 30)
        image = dataset.frame(camera, 0)
        for x, y in uv[valid]:
            cv2.circle(image, (int(x), int(y)), 1, (30, 30, 240), -1)
        cv2.imwrite(str(folder / f"{name}_floor_wall_boundary.png"), image)
        for pixel, point in zip(uv[valid], xy[valid]):
            observations.append({"camera": name, "uv": pixel.tolist(), "xy": point.tolist()})
    save_json(folder / "floor_wall_boundary_observations.json", observations)
    return observations


def ransac_lines(points, threshold=0.09, min_support=35, min_span=0.65, seed=7):
    rng = np.random.default_rng(seed)
    points = np.asarray(points, float)
    remaining = np.arange(len(points))
    result = []
    for _ in range(16):
        if len(remaining) < min_support:
            break
        best = np.zeros(len(remaining), bool)
        for _ in range(500):
            a, b = points[rng.choice(remaining, 2, replace=False)]
            v = b - a
            if np.linalg.norm(v) < 0.25:
                continue
            n = np.array([-v[1], v[0]]) / np.linalg.norm(v)
            mask = np.abs(points[remaining] @ n - a @ n) < threshold
            if mask.sum() > best.sum():
                best = mask
        if best.sum() < min_support:
            break
        indices = remaining[best]
        chosen = points[indices]
        center = np.median(chosen, 0)
        _, _, vt = np.linalg.svd(chosen - center, full_matrices=False)
        tangent = vt[0]
        normal = vt[1]
        if np.ptp(chosen @ tangent) >= min_span:
            result.append(
                {
                    "normal": normal.tolist(),
                    "position": float(normal @ center),
                    "support": len(indices),
                    "indices": indices.tolist(),
                    "span": float(np.ptp(chosen @ tangent)),
                }
            )
        remaining = remaining[~best]
    return result


def fit_boundary_room(observations, room_id, camera_centers, allow_open_ends=False):
    xy = np.array([o["xy"] for o in observations])
    lines = ransac_lines(xy)
    if len(lines) < 4:
        raise ValueError("Insufficient independent floor-wall boundary lines")
    weights = np.array([l["support"] * min(l["span"], 4) for l in lines])
    angles = np.arctan2(*np.array([l["normal"] for l in lines])[:, ::-1].T)
    candidates = np.deg2rad(np.linspace(-45, 45, 721))
    errors = np.abs((angles[:, None] - candidates[None, :] + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    yaw = float(
        candidates[
            np.argmax(
                np.sum(weights[:, None] * np.exp(-0.5 * (errors / np.deg2rad(6)) ** 2), axis=0)
            )
        ]
    )
    co, si = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[co, si], [-si, co]])
    rotated = xy @ rotation.T
    centers = np.asarray(camera_centers) @ rotation.T
    candidates = []
    for line in lines:
        normal = np.array(line["normal"]) @ rotation.T
        axis = int(np.argmax(abs(normal)))
        if np.rad2deg(np.arctan2(abs(normal[1 - axis]), abs(normal[axis]))) > 12:
            continue
        selected = rotated[line["indices"]]
        position = float(np.median(selected[:, axis]))
        residual = np.abs(selected[:, axis] - position)
        inliers = residual < 0.13
        if inliers.sum() < 25:
            continue
        cameras = sorted(
            {observations[i]["camera"] for i, ok in zip(line["indices"], inliers) if ok}
        )
        candidates.append(
            {
                "axis": axis,
                "position": position,
                "support": int(inliers.sum()),
                "cameras": cameras,
                "span": float(np.ptp(selected[inliers, 1 - axis])),
                "residual_median": float(np.median(residual[inliers])),
            }
        )
    evidence = []
    bounds = []
    for axis, side in [(0, -1), (0, 1), (1, -1), (1, 1)]:
        eligible = [c for c in candidates if c["axis"] == axis and side * c["position"] > 0.5]
        if not eligible:
            if not allow_open_ends:
                raise ValueError(f"Missing observed boundary side {axis}/{side}")
            position = float(np.percentile(rotated[:, axis], 97 if side > 0 else 3))
            bounds.append(position)
            evidence.append(
                {
                    "axis": axis,
                    "side": side,
                    "position": position,
                    "kind": "open_scene_limit",
                    "support": 0,
                    "status": "extent of visible seam observations; NOT an observed wall",
                }
            )
            continue

        # Rank observed line support, never the most extreme point/plane or padded extent.
        def score(line):
            enclosure = np.mean(side * (centers[:, axis] - line["position"]) <= 0.3)
            return line["support"] * min(line["span"], 4) * enclosure

        winner = max(eligible, key=score)
        bounds.append(winner["position"])
        evidence.append(dict(winner, side=side))
    xmin, xmax, ymin, ymax = bounds
    center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
    matrix = np.eye(3)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = -center
    p = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]) - center
    room = room_from_polygon(room_id, p)
    room.update(
        {
            "boundary_status": "RANSAC lines fitted to image floor-wall seams; Manhattan rectangular closure",
            "area_m2": float((xmax - xmin) * (ymax - ymin)),
            "wall_fit_evidence": evidence,
            "boundary_observation_count": len(observations),
            "dominant_yaw_deg": float(np.rad2deg(yaw)),
        }
    )
    for wall in room["walls"]:
        a, b = np.array(wall["start"]), np.array(wall["end"])
        normal = np.array(wall["outward_normal"])
        axis = int(np.argmax(abs(normal)))
        side = int(np.sign(normal[axis]))
        matching = next(e for e in evidence if e["axis"] == axis and e["side"] == side)
        wall["kind"] = matching.get("kind", "observed_wall")
    return room, matrix, {"ransac_lines": lines, "candidates": candidates, "selected": evidence}


def pinhole_pose(h, k):
    b = np.linalg.inv(k) @ np.linalg.inv(h)
    scale = 2 / (np.linalg.norm(b[:, 0]) + np.linalg.norm(b[:, 1]))
    result = None
    for sign in [1, -1]:
        r1, r2 = b[:, 0] * scale * sign, b[:, 1] * scale * sign
        u, _, vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
        r = u @ vt
        t = b[:, 2] * scale * sign
        c = -r.T @ t
        if c[2] > 0 and t[2] > 0:
            result = (r, t, c)
            break
    if result is None:
        raise ValueError("No physical camera pose")
    return result, b


def refine_mount_pose(entry, width, height, polygon, allow_cropped_principal=False):
    # H is image -> a right-handed metric map; keep it fixed to preserve person projections.
    h = np.asarray(entry["H_image_to_map"])
    initial = np.asarray(entry["K_estimated"])
    xmin, ymin = np.min(polygon, axis=0)
    xmax, ymax = np.max(polygon, axis=0)

    def evaluate(params):
        f, cx, cy = params
        k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        (r, t, c), b = pinhole_pose(h, k)
        u, v = b[:, 0], b[:, 1]
        constraints = [
            u @ v / (np.linalg.norm(u) * np.linalg.norm(v)),
            (np.linalg.norm(u) - np.linalg.norm(v)) / (np.linalg.norm(u) + np.linalg.norm(v)),
        ]
        distances = np.array([c[0] - xmin, xmax - c[0], c[1] - ymin, ymax - c[1]])
        # Soft mounting prior refines intrinsics, not a post-hoc translation of the camera icon.
        residual = [
            *np.array(constraints) * 200,
            (distances.min() - 0.12) * 2,
            (cx - width / 2) / width * 0.08,
            (cy - height / 2) / height * 0.08,
            (f - initial[0, 0]) / width * 0.03,
        ]
        residual.extend(np.minimum(distances, 0) * 10)
        return np.array(residual)

    lower = (
        [width * 0.28, -width, -height]
        if allow_cropped_principal
        else [width * 0.40, width * 0.20, 0]
    )
    upper = (
        [width * 4.5, width * 2, height * 2]
        if allow_cropped_principal
        else [width * 2, width * 0.80, height]
    )
    fit = least_squares(
        evaluate,
        [initial[0, 0], width / 2, height / 2],
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.2,
        max_nfev=500,
    )
    f, cx, cy = fit.x
    k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    (r, t, c), _ = pinhole_pose(h, k)
    direction = r.T @ np.array([0.0, 0.0, 1.0])
    hfov = float(np.arctan(cx / f) + np.arctan((width - cx) / f))
    distances = [c[0] - xmin, xmax - c[0], c[1] - ymin, ymax - c[1]]
    return {
        "K": k.tolist(),
        "R_world_to_camera": r.tolist(),
        "t_world_to_camera": t.tolist(),
        "camera_center_world": c.tolist(),
        "optical_axis_world": direction.tolist(),
        "horizontal_fov_rad": hfov,
        "wall_clearance_m": float(min(distances)),
        "status": "joint intrinsic/pose refinement with soft wall-mount prior; not measured extrinsics",
        "mount_prior_target_m": 0.12,
        "intrinsic_constraints": evaluate(fit.x)[:2].tolist(),
    }


def view_polygon(center, direction, hfov, room_polygon):
    from shapely.geometry import Polygon, LineString, Point

    center = np.asarray(center, float)
    axis = np.asarray(direction, float)[:2]
    axis /= np.linalg.norm(axis)
    yaw = np.arctan2(axis[1], axis[0])
    shape = Polygon(room_polygon)
    vertices = [center]
    for a in np.linspace(-hfov / 2, hfov / 2, 41):
        ray = np.array([np.cos(yaw + a), np.sin(yaw + a)])
        hit = shape.boundary.intersection(LineString([center, center + ray * 1000]))
        hits = []
        if not hit.is_empty:
            for part in getattr(hit, "geoms", [hit]):
                if isinstance(part, Point):
                    hits.append(np.array(part.coords[0]))
        positive = [p for p in hits if (p - center) @ ray > 1e-6]
        if positive:
            vertices.append(min(positive, key=lambda p: np.linalg.norm(p - center)))
    if len(vertices) < 3:
        raise ValueError("Camera rays do not intersect the observed room")
    return np.asarray(vertices)
