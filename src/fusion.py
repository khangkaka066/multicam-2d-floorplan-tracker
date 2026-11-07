"""One observation per camera per person; uncertainty-aware fusion and temporal global IDs."""

from copy import deepcopy
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from .geometry import project


def cosine_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.clip(1 - a @ b / (max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)), 0, 2))


def associate_frame(detections, transforms, gate=2.0):
    by_camera = {}
    for d in detections:
        if not np.isfinite(d["xy"]).all():
            continue
        by_camera.setdefault(d["camera"], []).append(d)
    if not by_camera:
        return []
    # Cardinality comes from detections, with no supplied room headcount.
    reference = max(
        sorted(by_camera),
        key=lambda c: (
            len(by_camera[c]),
            sum(d.get("ankle_confidence", 0) for d in by_camera[c]),
            c == "cam0",
        ),
    )
    base = by_camera[reference]
    axis = int(np.argmax(np.ptp(np.array([d["xy"] for d in base]), axis=0))) if len(base) > 1 else 1
    base = sorted(base, key=lambda d: d["xy"][axis])
    groups = [[d] for d in base]
    for camera in sorted(by_camera):
        if camera == reference:
            continue
        other = sorted(by_camera[camera], key=lambda d: d["xy"][axis])
        distances = np.array(
            [[np.linalg.norm(np.array(a["xy"]) - b["xy"]) for b in other] for a in base]
        )
        appearance = np.array(
            [[cosine_distance(a["appearance"], b["appearance"]) for b in other] for a in base]
        )
        ranks = np.abs(
            np.linspace(0, 1, len(base))[:, None] - np.linspace(0, 1, len(other))[None, :]
        )
        cost = distances + 0.35 * np.minimum(appearance / 0.20, 2) + 0.20 * ranks
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if distances[i, j] < gate and appearance[i, j] < 0.8:
                groups[i].append(other[j])
    return groups


def projection_jacobian(h, uv):
    h = np.asarray(h)
    u, v = uv
    w = h[2] @ [u, v, 1]
    if abs(w) < 1e-8:
        return np.full((2, 2), 1e6)
    numerator = h[:2] @ [u, v, 1]
    return (h[:2, :2] * w - numerator[:, None] * h[2, :2]) / w**2


def observation_weight(d, entry):
    camera = np.asarray(entry.get("camera_center_world", list(entry["camera_xy_estimate"]) + [2.0]))
    distance = float(np.linalg.norm(np.r_[d["xy"], 0.0] - camera))
    confidence = (
        float(d.get("ankle_confidence", 0))
        if d["method"] == "ankles"
        else float(d["confidence"]) * 0.15
    )
    if d.get("ankle_censored"):
        confidence *= 0.15
    jacobian = projection_jacobian(entry["H_image_to_map"], d["uv"])
    sensitivity = float(np.linalg.norm(jacobian, ord=2))
    conditioning = 1 / (1 + (sensitivity / 0.08) ** 2)
    person_confidence = float(d.get("detector_confidence", 1.0))
    truncation_penalty = 0.25 if d.get("lateral_truncation", False) else 1.0
    weight = (
        confidence
        / max(distance**2, 1.0)
        * conditioning
        * person_confidence**2
        * truncation_penalty
    )
    return max(weight, 1e-8), {
        "camera_distance_m": distance,
        "meters_per_pixel": sensitivity,
        "ankle_or_fallback_confidence": confidence,
        "conditioning_weight": conditioning,
        "person_confidence": person_confidence,
        "truncation_penalty": truncation_penalty,
    }


def robust_fuse(group, transforms, max_residual=0.35):
    positions = np.array([d["xy"] for d in group])
    weights = []
    for d in group:
        weight, detail = observation_weight(d, transforms["cameras"][d["camera"]])
        d["fusion_weight_components"] = detail
        d["fusion_weight"] = weight
        weights.append(weight)
    weights = np.asarray(weights)
    # Start from the most reliable camera, then accept a coherent neighborhood.
    center = positions[np.argmax(weights)].copy()
    for _ in range(6):
        residual = np.linalg.norm(positions - center, axis=1)
        robust = np.minimum(1.0, 0.20 / np.maximum(residual, 1e-6))
        updated = np.average(positions, axis=0, weights=weights * robust)
        if np.linalg.norm(updated - center) < 1e-5:
            break
        center = updated
    from itertools import combinations

    best = ()
    best_score = (-1, -1.0)
    for count in range(1, len(group) + 1):
        for subset in combinations(range(len(group)), count):
            selected = positions[list(subset)]
            diameter = np.max(np.linalg.norm(selected[:, None] - selected[None, :], axis=2))
            if diameter > max_residual:
                continue
            score = (len(subset), float(weights[list(subset)].sum()))
            if score > best_score:
                best, best_score = subset, score
    inliers = np.zeros(len(group), bool)
    inliers[list(best)] = True
    center = np.average(positions[inliers], axis=0, weights=weights[inliers])
    accepted = []
    for d, keep in zip(group, inliers):
        d["fusion_inlier"] = bool(keep)
        d["fusion_residual_m"] = float(np.linalg.norm(np.array(d["xy"]) - center))
        if keep:
            accepted.append(d)
    return center, accepted


def fuse_and_track(records, transforms, room, max_residual=0.35):
    polygon = np.asarray(room["polygon"], np.float32)
    tracks = []
    next_id = 1
    all_pair_errors = []
    for record in records:
        groups = associate_frame(record["detections"], transforms, gate=1.2)
        fused = []
        for group in groups:
            center, inliers = robust_fuse(group, transforms, max_residual)
            if len({d["camera"] for d in inliers}) < 2:
                continue
            clearance = float(cv2.pointPolygonTest(polygon, tuple(center), True))
            if clearance < 0.14:
                continue
            feature = np.mean([d["appearance"] for d in inliers], axis=0)
            feature /= max(np.linalg.norm(feature), 1e-9)
            pairs = [
                float(np.linalg.norm(np.array(a["xy"]) - b["xy"]))
                for i, a in enumerate(inliers)
                for b in inliers[i + 1 :]
            ]
            all_pair_errors.extend(pairs)
            fused.append(
                {
                    "xy": center.tolist(),
                    "room_id": room["id"],
                    "member_cameras": [d["camera"] for d in inliers],
                    "member_detection_ids": [d["detection_id"] for d in inliers],
                    "candidate_detection_ids": [d["detection_id"] for d in group],
                    "appearance": feature.tolist(),
                    "max_camera_residual_m": max(d["fusion_residual_m"] for d in inliers),
                    "max_pairwise_distance_m": max(pairs, default=0.0),
                    "boundary_clearance_m": clearance,
                }
            )
        active = [t for t in tracks if record["index"] - t["last_index"] <= 8]
        used = set()
        if active and fused:
            distance = np.array(
                [
                    [
                        np.linalg.norm(
                            np.array(t["xy"])
                            + np.array(t.get("velocity", [0.0, 0.0]))
                            * (record["index"] - t["last_index"])
                            - f["xy"]
                        )
                        for f in fused
                    ]
                    for t in active
                ]
            )
            appearance = np.array(
                [[cosine_distance(t["appearance"], f["appearance"]) for f in fused] for t in active]
            )
            rows, cols = linear_sum_assignment(distance + 0.30 * appearance)
            for i, j in zip(rows, cols):
                if distance[i, j] < 1.2:
                    fused[j]["id"] = active[i]["id"]
                    track = active[i]
                    gap = max(1, record["index"] - track["last_index"])
                    velocity = (np.array(fused[j]["xy"]) - track["xy"]) / gap
                    velocity = 0.5 * np.array(track.get("velocity", [0.0, 0.0])) + 0.5 * velocity
                    track.update(
                        xy=fused[j]["xy"],
                        appearance=fused[j]["appearance"],
                        last_index=record["index"],
                        velocity=velocity.tolist(),
                        hits=track.get("hits", 1) + 1,
                    )
                    used.add(j)
        for j, person in enumerate(fused):
            if j not in used:
                person["id"] = next_id
                tracks.append(
                    {
                        "id": next_id,
                        "xy": person["xy"],
                        "appearance": person["appearance"],
                        "last_index": record["index"],
                        "velocity": [0.0, 0.0],
                        "hits": 1,
                    }
                )
                next_id += 1
            for d in record["detections"]:
                if d["detection_id"] in person["candidate_detection_ids"]:
                    d["associated_id"] = person["id"]
                if d["detection_id"] in person["member_detection_ids"]:
                    d["fused_id"] = person["id"]
        observed_ids = {p["id"] for p in fused}
        for track in tracks:
            gap = record["index"] - track["last_index"]
            if track["id"] in observed_ids or not 1 <= gap <= 5 or track.get("hits", 1) < 3:
                continue
            prediction = np.array(track["xy"]) + np.array(track.get("velocity", [0.0, 0.0])) * gap
            clearance = float(cv2.pointPolygonTest(polygon, tuple(prediction), True))
            if clearance < 0.30:
                continue
            fused.append(
                {
                    "id": track["id"],
                    "xy": prediction.tolist(),
                    "room_id": room["id"],
                    "member_cameras": [],
                    "member_detection_ids": [],
                    "candidate_detection_ids": [],
                    "appearance": track["appearance"],
                    "boundary_clearance_m": clearance,
                    "state": "predicted_short_occlusion",
                    "predicted_frame_gap": gap,
                    "max_camera_residual_m": None,
                    "max_pairwise_distance_m": None,
                }
            )
        for person in fused:
            person.setdefault("state", "observed")
        record["fused_people"] = fused
        record["headcounts"] = {room["id"]: len(fused)}
    return {
        "track_ids_allocated": next_id - 1,
        "headcounts": {
            str(n): sum(r["headcounts"][room["id"]] == n for r in records)
            for n in sorted({r["headcounts"][room["id"]] for r in records})
        },
        "inlier_pair_distance_mean_m": float(np.mean(all_pair_errors)) if all_pair_errors else None,
        "inlier_pair_distance_p95_m": (
            float(np.percentile(all_pair_errors, 95)) if all_pair_errors else None
        ),
        "weight_rule": "ankle_confidence / camera_distance^2 times projection-conditioning penalty",
        "headcount_source": "multi-camera associations with >=2 inlier cameras; confirmed tracks may coast <=5 frames (0.20s), explicitly flagged; no hard-coded counts",
    }
