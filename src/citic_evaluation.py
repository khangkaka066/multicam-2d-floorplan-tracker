"""Independent comparison with CITIC's one-based, interpolated waypoint ground truth."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .geometry import project


def evaluate_transitions(predicted, truth, tolerance_seconds=4.):
    """One-to-one doorway-event comparison; timing tolerance does not alter inferred IDs."""
    cost = np.full((len(predicted), len(truth)), 1e6)
    for i, a in enumerate(predicted):
        for j, b in enumerate(truth):
            if a["from_room"] == b["from_room"] and a["to_room"] == b["to_room"]:
                cost[i, j] = abs(a["source_frame"] - b["source_frame"]) / 25.
    rows, cols = linear_sum_assignment(cost)
    matches = [dict(predicted_id=predicted[i]["id"], gt_id=truth[j]["gt_id"],
                    from_room=predicted[i]["from_room"], to_room=predicted[i]["to_room"],
                    predicted_source_frame=predicted[i]["source_frame"],
                    gt_source_frame=truth[j]["source_frame"], timing_error_seconds=float(cost[i, j]))
               for i, j in zip(rows, cols) if cost[i, j] <= tolerance_seconds]
    return dict(tolerance_seconds=tolerance_seconds, predicted_events=len(predicted),
                gt_events=len(truth), matched_events=len(matches),
                precision=len(matches) / max(len(predicted), 1),
                recall=len(matches) / max(len(truth), 1),
                identity_correctness="evaluated separately as localization-gated identity switches",
                matches=matches)


def gt_at_frame(gt, source_frame, matrix):
    rows = gt[gt[:, 0] == source_frame + 1]
    xy = project(matrix, rows[:, 7:9])
    return [(int(row[1]), point.tolist()) for row, point in zip(rows, xy)]


def evaluate(records, gt, config):
    errors, matches, totals = [], [], [0, 0]
    gt_transitions = []
    previous_rooms = {}
    from shapely.geometry import Point, Polygon
    shapes = {r["id"]: Polygon(r["polygon"]) for r in config["rooms"]}
    for row in gt:
        xy = project(config["T_source_to_map"], [row[7:9]])[0]
        room = next((name for name, shape in shapes.items() if shape.covers(Point(xy))), None)
        if room:
            identity = int(row[1])
            old = previous_rooms.get(identity)
            if old and old != room:
                gt_transitions.append(dict(gt_id=identity, from_room=old, to_room=room,
                                           source_frame=int(row[0]) - 1))
            previous_rooms[identity] = room
    for record in records:
        truth = gt_at_frame(gt, record["source_frame"], config["T_source_to_map"])
        people = record["fused_people"]
        totals[0] += len(truth)
        totals[1] += len(people)
        if not truth or not people:
            continue
        distances = np.linalg.norm(np.array([p["xy"] for p in people])[:, None] -
                                   np.array([p[1] for p in truth])[None], axis=2)
        rows, cols = linear_sum_assignment(distances)
        for i, j in zip(rows, cols):
            error = float(distances[i, j])
            match = dict(source_frame=record["source_frame"], predicted_id=people[i]["id"],
                         gt_id=truth[j][0], error_m=error, room_id=people[i]["room_id"])
            matches.append(match)
            if error <= 2.:
                errors.append(error)
    at_one = sum(m["error_m"] <= 1. for m in matches)
    at_two = sum(m["error_m"] <= 2. for m in matches)
    # Report identity switches under a 2m localization gate; GT IDs never enter the tracker.
    last = {}
    switches = 0
    for m in matches:
        if m["error_m"] > 2.:
            continue
        if m["gt_id"] in last and last[m["gt_id"]] != m["predicted_id"]:
            switches += 1
        last[m["gt_id"]] = m["predicted_id"]
    report = dict(ground_truth_positions=totals[0], predicted_positions=totals[1],
                  matched_positions=at_two, matches_at_1m=at_one,
                  mean_error_m=float(np.mean(errors)) if errors else None,
                  median_error_m=float(np.median(errors)) if errors else None,
                  p95_error_m=float(np.percentile(errors, 95)) if errors else None,
                  recall_at_1m=at_one / max(totals[0], 1),
                  precision_at_1m=at_one / max(totals[1], 1),
                  recall_at_2m=at_two / max(totals[0], 1),
                  precision_at_2m=at_two / max(totals[1], 1),
                  identity_switches_at_2m=switches,
                  gt_input_to_inference=False, metric_scope="Hungarian one-to-one frame matches; 2m error gate",
                  gt_limitations="phone waypoint interpolation, not per-frame surveyed footpoints; capture1 excludes invitado1 in author GT script",
                  ground_truth_room_transitions=gt_transitions)
    return report, matches
