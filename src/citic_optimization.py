"""Bounded overlap-only translation refinement, validated on later frames without GT."""
from copy import deepcopy
import numpy as np
from scipy.optimize import linear_sum_assignment

from .fusion import cosine_distance
from .geometry import project


def refine_overlap(records, transforms):
    pairs = []
    for record in records:
        a = [d for d in record["detections"] if d["camera"] == "HallWide" and d["method"] == "ankles" and d["foreground_supported"]]
        b = [d for d in record["detections"] if d["camera"] == "HallSeg" and d["method"] == "ankles" and d["foreground_supported"]]
        if not a or not b:
            continue
        cost = np.array([[np.linalg.norm(np.array(x["xy"]) - y["xy"]) for y in b] for x in a])
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if (cost[i, j] < 0.65 and cosine_distance(a[i]["appearance"], b[j]["appearance"]) < 0.65
                    and a[i].get("floor_supported", True) and b[j].get("floor_supported", True)):
                pairs.append((record["source_frame"], np.array(a[i]["xy"]) - b[j]["xy"]))
    boundary = records[len(records) // 2]["source_frame"]
    train = np.array([p for f, p in pairs if f < boundary]).reshape(-1, 2)
    heldout = np.array([p for f, p in pairs if f >= boundary]).reshape(-1, 2)
    report = dict(method="HallSeg translation toward HallWide simultaneous pose consensus",
                  gt_used=False, train_before_source_frame=boundary,
                  training_pairs=len(train), validation_pairs=len(heldout), accepted=False,
                  maximum_translation_m=0.5, note="Ground truth is excluded; independent single-camera rooms retain published calibration")
    result = deepcopy(transforms)
    if len(train) >= 8 and len(heldout) >= 5:
        shift = np.median(train, axis=0)
        before = np.linalg.norm(heldout, axis=1)
        after = np.linalg.norm(heldout - shift, axis=1)
        report.update(candidate_translation_m=shift.tolist(),
                      heldout_mean_before_m=float(before.mean()), heldout_mean_after_m=float(after.mean()))
        if np.linalg.norm(shift) <= 0.5 and after.mean() < before.mean():
            correction = np.eye(3)
            correction[:2, 2] = shift
            e = result["cameras"]["HallSeg"]
            e["H_image_to_map_before_overlap"] = e["H_image_to_map"]
            e["overlap_correction"] = correction.tolist()
            h = correction @ e["H_image_to_map"]
            e["H_image_to_map"] = h.tolist()
            e["H_map_to_image"] = np.linalg.inv(h).tolist()
            report["accepted"] = True
    return result, report
