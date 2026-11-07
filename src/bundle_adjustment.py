"""Joint projective footpoint refinement with fixed architectural geometry and held-out checks."""

from copy import deepcopy
import cv2
import numpy as np
from scipy.optimize import least_squares
from .geometry import project
from .fusion import associate_frame, observation_weight


def geometric_median(points, weights):
    center = np.average(points, axis=0, weights=weights)
    for _ in range(40):
        w = weights / np.maximum(np.linalg.norm(points - center, axis=1), 0.01)
        updated = np.average(points, axis=0, weights=w)
        if np.linalg.norm(updated - center) < 1e-6:
            break
        center = updated
    return center


def refine_homographies(records, transforms, room):
    cameras = sorted(transforms["cameras"])
    camera_index = {c: i for i, c in enumerate(cameras)}
    polygon = np.asarray(room["polygon"])
    lower, upper = polygon.min(0), polygon.max(0)
    span = upper - lower
    origin = (upper + lower) / 2
    normalizer = np.array(
        [
            [1 / span[0], 0, -origin[0] / span[0]],
            [0, 1 / span[1], -origin[1] / span[1]],
            [0, 0, 1.0],
        ]
    )
    inverse = np.linalg.inv(normalizer)
    observations = []
    groups = []
    for record in records:
        reliable = [
            d
            for d in record["detections"]
            if d["method"] == "ankles"
            and d.get("ankle_confidence", 0) >= 0.5
            and not d.get("ankle_censored", False)
        ]
        for group in associate_frame(reliable, transforms, gate=2.0):
            selected = [
                d
                for d in group
                if d["method"] == "ankles"
                and d.get("ankle_confidence", 0) >= 0.5
                and not d.get("ankle_censored", False)
            ]
            selected = [
                d
                for d in selected
                if cv2.pointPolygonTest(np.asarray(polygon, np.float32), tuple(d["xy"]), True)
                > -0.3
            ]
            if len(selected) < 2:
                continue
            positions = np.array([d["xy"] for d in selected])
            weights = np.array(
                [observation_weight(d, transforms["cameras"][d["camera"]])[0] for d in selected]
            )
            weights = np.maximum(weights / weights.sum(), 0.025)
            weights /= weights.sum()
            target = geometric_median(positions, weights)
            start = len(observations)
            for d, w in zip(selected, weights):
                observations.append((camera_index[d["camera"]], d["xy"], w, d))
            groups.append(
                {
                    "indices": list(range(start, len(observations))),
                    "index": record["index"],
                    "target": target,
                }
            )
    if len(groups) < 20:
        return deepcopy(transforms), {
            "status": "insufficient multi-view ankle evidence",
            "groups": len(groups),
        }
    base = np.array([x[1] for x in observations])
    indices = np.array([x[0] for x in observations])
    weights = np.array([x[2] for x in observations])
    normalized = project(normalizer, base)
    homogeneous = np.c_[normalized, np.ones(len(base))]
    controls = np.array(
        [
            [x, y]
            for x in np.linspace(lower[0], upper[0], 3)
            for y in np.linspace(lower[1], upper[1], 3)
        ]
    )
    control_norm = np.c_[project(normalizer, controls), np.ones(len(controls))]
    validation_indices = {r["index"] for r in records if r["index"] % 5 == 3}
    training = []
    for original_group in groups:
        if original_group["index"] in validation_indices:
            continue
        ids = [
            i
            for i in original_group["indices"]
            if observations[i][3].get("detector_confidence", 1.0) >= 0.65
            and not observations[i][3].get("lateral_truncation", False)
        ]
        if len(ids) < 2:
            continue
        weights[ids] /= weights[ids].sum()
        training.append(
            dict(original_group, indices=ids, target=geometric_median(base[ids], weights[ids]))
        )
    heldout = [g for g in groups if g["index"] in validation_indices]

    def corrections(parameters):
        p = parameters.reshape(len(cameras), 8)
        matrix = np.tile(np.eye(3), (len(cameras), 1, 1))
        matrix[:, 0, :] = np.c_[1 + p[:, 0], p[:, 1], p[:, 2]]
        matrix[:, 1, :] = np.c_[p[:, 3], 1 + p[:, 4], p[:, 5]]
        matrix[:, 2, :2] = p[:, 6:8]
        return matrix

    def warp(parameters):
        matrices = corrections(parameters)
        out = np.einsum("nij,nj->ni", matrices[indices], homogeneous)
        return out[:, :2] / np.maximum(out[:, 2:3], 0.15) * span + origin

    pair_i = []
    pair_j = []
    pair_w = []
    train_group_ids = []
    for g in training:
        ids = g["indices"]
        train_group_ids.append(ids)
        for n, i in enumerate(ids):
            for j in ids[n + 1 :]:
                pair_i.append(i)
                pair_j.append(j)
                pair_w.append(np.sqrt(weights[i] * weights[j]))
    pair_i = np.array(pair_i)
    pair_j = np.array(pair_j)
    pair_w = np.array(pair_w)
    train_obs = np.array([i for g in training for i in g["indices"]])
    train_group = np.array([n for n, g in enumerate(training) for _ in g["indices"]])
    train_targets = np.array([g["target"] for g in training])

    def residual(parameters):
        predicted = warp(parameters)
        matrices = corrections(parameters)
        values = [((predicted[pair_i] - predicted[pair_j]) * pair_w[:, None]).ravel()]
        centroids = np.zeros((len(training), 2))
        np.add.at(centroids, train_group, predicted[train_obs] * weights[train_obs, None])
        values.append(((centroids - train_targets) * 0.35).ravel())
        # Preserve room-scale gauge and prevent all homographies collapsing into one point.
        control = np.einsum("cij,nj->cni", matrices, control_norm)
        control_xy = control[:, :, :2] / control[:, :, 2:3] * span + origin
        values.append(((control_xy - controls[None, :, :]) * 0.05).ravel())
        displacement = np.linalg.norm(control_xy - controls[None, :, :], axis=2)
        values.append((np.maximum(displacement - 1.25, 0) ** 2 * 10).ravel())
        values.append(((control_xy.mean(0) - controls) * 0.05).ravel())
        values.append((np.minimum(predicted[train_obs] - lower + 0.05, 0) * 0.6).ravel())
        values.append((np.maximum(predicted[train_obs] - upper - 0.05, 0) * 0.6).ravel())
        # Smooth finite projective maps over the architectural domain.
        determinants = np.linalg.det(matrices[:, :2, :2])
        values.append(np.maximum(0.50 - determinants, 0) * 4)
        values.append(np.maximum(0.35 - control[:, :, 2], 0).ravel() * 4)
        return np.concatenate(values)

    zero = np.zeros(8 * len(cameras))
    single_lower = np.array([-0.25, -0.12, -1 / span[0], -0.12, -0.25, -1 / span[1], -0.20, -0.55])
    single_upper = -single_lower
    fitted = least_squares(
        residual,
        zero,
        bounds=(np.tile(single_lower, len(cameras)), np.tile(single_upper, len(cameras))),
        loss="soft_l1",
        f_scale=0.08,
        max_nfev=180,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )

    def errors(parameters, subset):
        predicted = warp(parameters)
        distances = []
        for g in subset:
            ids = g["indices"]
            distances.extend(
                np.linalg.norm(predicted[ids[i]] - predicted[ids[j]])
                for i in range(len(ids))
                for j in range(i + 1, len(ids))
            )
        return np.array(distances)

    before = errors(zero, heldout)
    after = errors(fitted.x, heldout)
    accept = bool(
        len(after) > 10
        and np.mean(after) < np.mean(before) * 0.98
        and np.percentile(after, 90) < np.percentile(before, 90)
    )
    parameters = fitted.x if accept else zero
    result = deepcopy(transforms)
    matrices = corrections(parameters)
    for camera, matrix in zip(cameras, matrices):
        entry = result["cameras"][camera]
        correction = inverse @ matrix @ normalizer
        h = correction @ np.asarray(entry["H_image_to_map"])
        entry["H_image_to_map_before_bundle"] = deepcopy(entry["H_image_to_map"])
        entry["H_image_to_map"] = h.tolist()
        entry["H_map_to_image"] = np.linalg.inv(h).tolist()
        entry["footpoint_bundle_correction"] = correction.tolist()
        entry["projection_refinement_status"] = (
            "joint 8-parameter projective correction; architectural pose retained"
        )
    metrics = {
        "status": (
            "accepted" if accept else "heldout improvement insufficient; original mapping retained"
        ),
        "unobserved_control_displacement_prior_m": 1.25,
        "training_person_confidence_min": 0.65,
        "exclude_laterally_truncated_training_poses": True,
        "validation_scope": "all reliable-ankle pairs, including low person-confidence cases excluded from training",
        "method": "joint 8-DOF corrections for all cameras; robust pairwise L2, distance/confidence weights, room constraints, gauge/control priors",
        "training_groups": len(training),
        "heldout_groups": len(heldout),
        "validation_split": "interleaved offline calibration, index % 5 == 3 withheld; not a future-time test",
        "validation_frame_indices": sorted(validation_indices),
        "requested_frame_78_used_for_fit": any(r["index"] == 78 for r in records)
        and 78 not in validation_indices,
        "validation_pixels_used_for_boundary_loss": False,
        "optimizer_evaluations": int(fitted.nfev),
        "near_camera_targets": "uncertainty-weighted geometric medians of original ankle observations; not ground truth",
        "heldout_pair_mean_before_m": float(before.mean()),
        "heldout_pair_mean_after_m": float(after.mean()),
        "heldout_pair_p95_before_m": float(np.percentile(before, 95)),
        "heldout_pair_p95_after_m": float(np.percentile(after, 95)),
        "heldout_pairs_under_035_before_fraction": float(np.mean(before < 0.35)),
        "heldout_pairs_under_035_after_fraction": float(np.mean(after < 0.35)),
        "accepted": accept,
        "max_control_displacement_m": float(
            max(
                np.linalg.norm(
                    project(inverse @ matrix @ normalizer, controls) - controls, axis=1
                ).max()
                for matrix in matrices
            )
        ),
        "target_met_for_all_heldout_pairs": bool(after.max() < 0.35),
    }
    return result, metrics
