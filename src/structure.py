"""Metric-depth reconstruction and structural RANSAC; no trajectory-derived boundaries."""

from pathlib import Path
import hashlib
import json
import sys
import cv2
import numpy as np
from scipy.optimize import minimize_scalar, least_squares
from .geometry import project, save_json

ROOT = Path(__file__).resolve().parents[1]


def fit_plane(points, threshold=0.08, iterations=400, hint=None, vertical=False, seed=3):
    points = np.asarray(points, float)
    rng = np.random.default_rng(seed)
    if len(points) < 30:
        raise ValueError("Insufficient points for a structural plane")
    best = np.zeros(len(points), bool)
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n)
        if length < 1e-7:
            continue
        n /= length
        if vertical and abs(n[2]) > 0.20:
            continue
        if hint is not None and abs(n @ hint) < np.cos(np.deg2rad(35)):
            continue
        inliers = np.abs(points @ n - a @ n) < threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 30:
        raise ValueError("No supported structural plane")
    selected = points[best]
    center = selected.mean(0)
    _, _, vt = np.linalg.svd(selected - center, full_matrices=False)
    n = vt[-1]
    if hint is not None and n @ hint < 0:
        n = -n
    if vertical and abs(n[2]) > 0.25:
        raise ValueError("Refined wall plane is not vertical")
    d = -n @ center
    inliers = np.abs(points @ n + d) < threshold
    return n, float(d), inliers


def rotation_between(a, b):
    a = np.asarray(a) / np.linalg.norm(a)
    b = np.asarray(b) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.clip(a @ b, -1, 1)
    if c < -0.99999:
        raise ValueError("Opposite gravity directions")
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def camera_from_homography(entry, width, height, allow_free_principal=False):
    # EPFL top-view Y points down; reflect Y before constructing a right-handed Z-up frame.
    h = np.asarray(entry["H_map_to_image"]) @ np.diag([1.0, -1.0, 1.0])
    principal = np.array([[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1.0]]) @ h

    def cost(focal):
        b = np.diag([1 / focal, 1 / focal, 1]) @ principal
        u, v = b[:, 0], b[:, 1]
        return (u @ v / (np.linalg.norm(u) * np.linalg.norm(v))) ** 2 + (
            (np.linalg.norm(u) - np.linalg.norm(v)) / (np.linalg.norm(u) + np.linalg.norm(v))
        ) ** 2

    fit = minimize_scalar(cost, bounds=(width * 0.4, width * 3), method="bounded")
    f = float(fit.x)
    k = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
    if allow_free_principal and fit.fun > 0.0001:

        def residual(params):
            focal, cx, cy = params
            basis = np.linalg.inv([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]]) @ h
            a, b = basis[:, 0], basis[:, 1]
            return [
                a @ b / (np.linalg.norm(a) * np.linalg.norm(b)) * 100,
                (np.linalg.norm(a) - np.linalg.norm(b))
                / (np.linalg.norm(a) + np.linalg.norm(b))
                * 100,
                (cx - width / 2) / width * 0.001,
                (cy - height / 2) / height * 0.001,
            ]

        solved = least_squares(
            residual,
            [f, width / 2, height / 2],
            bounds=([width * 0.28, -width, -height], [width * 4.5, width * 2, height * 2]),
            max_nfev=1000,
        )
        f, cx, cy = solved.x
        k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        fit.fun = float(np.sum(np.square(residual(solved.x)[:2])) / 10000)
    b = np.linalg.inv(k) @ h
    scale = 2 / (np.linalg.norm(b[:, 0]) + np.linalg.norm(b[:, 1]))
    for sign in [1, -1]:
        r1, r2 = b[:, 0] * scale * sign, b[:, 1] * scale * sign
        r = np.column_stack([r1, r2, np.cross(r1, r2)])
        u, _, vt = np.linalg.svd(r)
        r = u @ vt
        t = b[:, 2] * scale * sign
        center = -r.T @ t
        if center[2] > 0:
            break
    if center[2] <= 0 or fit.fun > 0.01:
        raise ValueError("Ground calibration incompatible with centered square-pixel pinhole model")
    return k, r, t, center, float(fit.fun)


class MetricDepthBackend:
    def __init__(self, cache):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.checkpoint = ROOT / "models/depth_anything_v2_metric_hypersim_vits.pth"
        self.vendor = ROOT / "models/Depth-Anything-V2/metric_depth"
        if not self.checkpoint.exists() or not self.vendor.exists():
            raise FileNotFoundError("Run python scripts/download_structural_models.py first")
        self.weight_hash = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()

    def infer(self, image, name):
        key = hashlib.sha256(image.tobytes()).hexdigest()
        meta_path = self.cache / f"{name}_depth_source.json"
        path = self.cache / f"{name}_depth.npy"
        metadata = {"image_sha256": key, "weights_sha256": self.weight_hash, "input_size": 518}
        if meta_path.exists() and path.exists() and json.loads(meta_path.read_text()) == metadata:
            return np.load(path)
        if self.model is None:
            import torch

            sys.path.insert(0, str(self.vendor))
            from depth_anything_v2.dpt import DepthAnythingV2

            self.model = DepthAnythingV2(
                encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20
            )
            self.model.load_state_dict(
                torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            )
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model.to(device).eval()
        depth = self.model.infer_image(image, 518)
        np.save(path, depth)
        save_json(meta_path, metadata)
        cv2.imwrite(
            str(self.cache / f"{name}_depth.png"),
            cv2.applyColorMap(np.uint8(np.clip(depth / 10, 0, 1) * 255), cv2.COLORMAP_TURBO),
        )
        return depth


def reconstruct(dataset, transforms, output):
    folder = Path(output) / "structure"
    folder.mkdir(parents=True, exist_ok=True)
    backend = MetricDepthBackend(folder)
    from .semantics import SemanticBackend

    semantics = SemanticBackend(folder)
    views = []
    for camera in dataset.cameras:
        image = dataset.frame(camera, 0)
        depth = backend.infer(image, camera["id"])
        labels = semantics.infer(image, camera["id"])
        height, width = depth.shape
        k, r, t, center, error = camera_from_homography(
            transforms["cameras"][camera["id"]],
            width,
            height,
            allow_free_principal=bool(dataset.manifest.get("plane_rectification_file")),
        )
        yy, xx = np.mgrid[:height, :width]
        rays = np.stack(
            [(xx - k[0, 2]) / k[0, 0], (yy - k[1, 2]) / k[1, 1], np.ones_like(xx)], axis=-1
        )
        points = rays * depth[..., None]
        # Carpet is used ONLY to seed the ground plane, never as the room outline.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        seed = (labels == 3) | (labels == 28)
        seed[: int(height * 0.48)] = False
        n, d, inliers = fit_plane(points[seed][::3], hint=r[:, 2], threshold=0.08)
        if d < 0:
            n, d = -n, -d
        if n @ r[:, 2] < 0:
            raise ValueError("Floor gravity sign inconsistent")
        scale = d / center[2]
        views.append(
            dict(
                camera=camera,
                image=image,
                depth=depth,
                points=points,
                k=k,
                r=r,
                t=t,
                center=center,
                floor_n=n,
                floor_d=d,
                scale=scale,
                seed=seed,
                intrinsics_error=error,
                floor_ratio=float(inliers.mean()),
                labels=labels,
            )
        )
        print(
            camera["id"],
            "estimated focal",
            round(k[0, 0], 1),
            "floor height",
            round(d, 3),
            "m; m/unit",
            round(scale, 5),
            flush=True,
        )
    meters_per_unit = (
        1.0
        if dataset.manifest.get("coordinate_units") == "meters"
        else float(np.median([v["scale"] for v in views]))
    )
    point_sets = []
    wall_sets = []
    planes = []
    details = {}
    for view in views:
        camera = view["camera"]
        name = camera["id"]
        r = view["r"]
        gravity = rotation_between(view["floor_n"], r[:, 2])
        correction = view["center"][2] * meters_per_unit / view["floor_d"]
        pc = view["points"] @ gravity.T * correction
        world = (pc - view["t"] * meters_per_unit) @ r
        # Align the depth floor to known floor observations with a rigid XY transform.
        ys, xs = np.where(view["seed"])
        source = world[ys[::8], xs[::8], :2]
        target = (
            project(transforms["cameras"][name]["H_image_to_map"], np.c_[xs[::8], ys[::8]])
            * meters_per_unit
            * np.array([1.0, -1.0])
        )
        a, b = source - source.mean(0), target - target.mean(0)
        u, _, vt = np.linalg.svd(a.T @ b)
        rot = u @ vt
        if np.linalg.det(rot) < 0:
            u[:, -1] *= -1
            rot = u @ vt
        shift = target.mean(0) - source.mean(0) @ rot
        world[..., :2] = world[..., :2] @ rot + shift
        normals = np.cross(np.gradient(world, axis=1), np.gradient(world, axis=0))
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-9)
        hsv = cv2.cvtColor(view["image"], cv2.COLOR_BGR2HSV)
        # Light wall surfaces, vertical local normals, height slicing: remove floors/tables/chairs.
        wall_mask = (
            (view["labels"] == 0)
            & (np.abs(normals[:, :, 2]) < 0.30)
            & (world[:, :, 2] >= 0.1)
            & (world[:, :, 2] <= 2.2)
        )
        wall_mask[:3] = False
        wall_mask[-3:] = False
        wall_mask[:, :3] = False
        wall_mask[:, -3:] = False
        candidates = world[wall_mask][::2]
        accepted = []
        remaining = candidates.copy()
        for _ in range(5):
            try:
                n, d, mask = fit_plane(remaining, threshold=0.10, vertical=True, iterations=300)
            except ValueError:
                break
            selected = remaining[mask]
            remaining = remaining[~mask]
            tangent = np.array([-n[1], n[0], 0])
            span = np.ptp(selected @ tangent)
            zspan = float(np.percentile(selected[:, 2], 95) - np.percentile(selected[:, 2], 5))
            if len(selected) < 80 or span < 0.9 or zspan < 0.65:
                continue
            error = float(np.median(np.abs(selected @ n + d)))
            plane = {
                "id": f"{name}_wall{len(accepted)}",
                "camera": name,
                "normal": n.tolist(),
                "offset": d,
                "support": len(selected),
                "span_m": float(span),
                "height_span_m": zspan,
                "median_error_m": error,
            }
            planes.append(plane)
            accepted.append(selected)
            wall_sets.append(selected)
        point_sets.append(world[::3, ::3].reshape(-1, 3))
        camera_center = view["center"][:2] * meters_per_unit @ rot + shift
        details[name] = {
            "K": view["k"].tolist(),
            "R_world_to_camera": r.tolist(),
            "t_world_to_camera": (view["t"] * meters_per_unit).tolist(),
            "camera_xy_m": camera_center.tolist(),
            "camera_height_m": float(view["center"][2] * meters_per_unit),
            "intrinsics_status": "estimated from ground homography; square pixels, zero skew; principal point may be free for projective-plane rectification",
            "intrinsics_constraint_error": view["intrinsics_error"],
            "floor_plane_camera": [*view["floor_n"].tolist(), view["floor_d"]],
            "floor_inlier_ratio": view["floor_ratio"],
            "depth_scale_correction": correction,
            "gravity_correction_deg": float(
                np.rad2deg(np.arccos(np.clip(view["floor_n"] @ r[:, 2], -1, 1)))
            ),
            "floor_registration_median_m": float(
                np.median(np.linalg.norm(source @ rot + shift - target, axis=1))
            ),
            "wall_planes": len(accepted),
        }
        print(
            name, "structural walls", len(accepted), "candidate points", len(candidates), flush=True
        )
        np.savez_compressed(
            folder / f"{name}_cloud.npz",
            points=world.astype("float32"),
            wall_mask=wall_mask,
            floor_seed=view["seed"],
        )
    if len(planes) < 4:
        raise ValueError(
            f"Only {len(planes)} wall planes supported; refusing to invent a room boundary"
        )
    walls = np.vstack(wall_sets)
    cloud = np.vstack(point_sets)
    np.savez_compressed(
        folder / "structural_cloud.npz",
        points=cloud.astype("float32"),
        wall_points=walls.astype("float32"),
    )
    report = {
        "model": "Depth Anything V2 Metric Hypersim Small",
        "weights_sha256": backend.weight_hash,
        "meters_per_epfl_unit_estimate": meters_per_unit,
        "metric_status": "model-estimated, not survey validated",
        "wall_height_slice_m": [0.1, 2.2],
        "cameras": details,
        "planes": planes,
    }
    save_json(folder / "reconstruction.json", report)
    return walls, cloud, report
