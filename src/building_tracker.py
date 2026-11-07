"""Observation-driven metric tracking across a connected building; no GT inputs."""
from copy import deepcopy
import heapq

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, Polygon

from .fusion import robust_fuse, cosine_distance


def associate_building_frame(detections, gate=0.85):
    """Keep every unmatched observation from every camera in disjoint spaces."""
    by_camera = {}
    for detection in detections:
        by_camera.setdefault(detection["camera"], []).append(detection)
    groups = []
    for camera, candidates in sorted(by_camera.items()):
        used = set()
        if groups:
            centers = [np.mean([d["xy"] for d in g], axis=0) for g in groups]
            distance = np.array([[np.linalg.norm(p - d["xy"]) for d in candidates] for p in centers])
            appearance = np.array([[cosine_distance(g[0]["appearance"], d["appearance"])
                                    for d in candidates] for g in groups])
            rows, cols = linear_sum_assignment(distance + 0.4 * appearance)
            for i, j in zip(rows, cols):
                if distance[i, j] < gate and appearance[i, j] < 0.65:
                    groups[i].append(candidates[j])
                    used.add(j)
        groups.extend([[d] for j, d in enumerate(candidates) if j not in used])
    return groups


class BuildingTracker:
    def __init__(self, config, transforms):
        self.config, self.transforms = config, transforms
        self.rooms = {r["id"]: Polygon(r["polygon"]) for r in config["rooms"]}
        self.doors = {}
        for link in config["alignment"]["connections"]:
            a, b = link["room_a"], link["room_b"]
            room = next(r for r in config["rooms"] if r["id"] == a)
            door = next(d for d in room["doors"] if d["id"] == link["door_a"])
            self.doors[frozenset([a, b])] = np.array(door["center"])
        self.tracks = []
        self.transitions = []
        self.next_id = 1

    def room_at(self, xy):
        return next((name for name, shape in self.rooms.items() if shape.covers(Point(xy))), None)

    def travel_distance(self, a, b):
        """Shortest path via shared doorway centers; prohibit paths through solid walls."""
        if a["room_id"] == b["room_id"]:
            return float(np.linalg.norm(np.array(a["xy"]) - b["xy"]))
        queue = [(0., a["room_id"], tuple(a["xy"]), ())]
        while queue:
            distance, room, point, visited = heapq.heappop(queue)
            if room == b["room_id"]:
                return distance + float(np.linalg.norm(np.array(point) - b["xy"]))
            for edge, center in self.doors.items():
                if room not in edge:
                    continue
                neighbor = next(n for n in edge if n != room)
                if neighbor in visited:
                    continue
                step = float(np.linalg.norm(np.array(point) - center))
                heapq.heappush(queue, (distance + step, neighbor, tuple(center), (*visited, room)))
        return float("inf")

    def update(self, record):
        valid = [d for d in record["detections"] if d["display_accepted"]]
        groups = associate_building_frame(valid)
        fused = []
        for group in groups:
            xy, inliers = robust_fuse(group, self.transforms, max_residual=0.65)
            # Retain singleton outliers as independent observations, so fusion cannot erase people.
            subsets = [inliers] + [[d] for d in group if d not in inliers]
            for members in subsets:
                if not members:
                    continue
                weights = [d["fusion_weight"] for d in members]
                xy = np.average([d["xy"] for d in members], axis=0, weights=weights)
                room = self.room_at(xy)
                if room is None:
                    continue
                feature = np.mean([d["appearance"] for d in members], axis=0)
                feature /= max(np.linalg.norm(feature), 1e-9)
                fused.append(dict(xy=xy.tolist(), room_id=room, appearance=feature.tolist(),
                                  member_cameras=[d["camera"] for d in members],
                                  member_detection_ids=[d["detection_id"] for d in members],
                                  state="observed", observation_count=len(members)))
        time = record["time_seconds"]
        active = [t for t in self.tracks if time - t["time_seconds"] <= 30.]
        used = set()
        if active and fused:
            cost = np.full((len(active), len(fused)), 1e6)
            for i, track in enumerate(active):
                dt = max(0.04, time - track["time_seconds"])
                for j, person in enumerate(fused):
                    distance = self.travel_distance(track, person)
                    gallery = track.get("appearance_gallery", [track["appearance"]])
                    appearance = min(cosine_distance(f, person["appearance"]) for f in gallery)
                    if distance <= 0.75 + 2.5 * dt and appearance < 0.60:
                        predicted = np.array(track["xy"]) + np.array(track.get("velocity", [0., 0.])) * min(dt, 1.)
                        residual = np.linalg.norm(predicted - person["xy"])
                        cost[i, j] = residual / (0.75 + 2.5 * dt) + 2.0 * appearance + 0.01 * dt
            rows, columns = linear_sum_assignment(cost)
            for i, j in zip(rows, columns):
                if cost[i, j] >= 1.8:
                    continue
                person, track = fused[j], active[i]
                person["id"] = track["id"]
                if person["room_id"] != track["room_id"]:
                    self.transitions.append(dict(id=track["id"], from_room=track["room_id"],
                                                 to_room=person["room_id"],
                                                 source_frame=record["source_frame"],
                                                 time_seconds=time, previous_time_seconds=track["time_seconds"],
                                                 doorway_path_distance_m=self.travel_distance(track, person),
                                                 appearance_distance=cosine_distance(track["appearance"], person["appearance"])))
                dt = max(time - track["time_seconds"], 0.04)
                velocity = (np.array(person["xy"]) - track["xy"]) / dt
                person["velocity"] = (0.5 * velocity + 0.5 * np.array(track.get("velocity", [0., 0.]))).tolist()
                gallery = [*track.get("appearance_gallery", [track["appearance"]]), person["appearance"]][-8:]
                track.update(deepcopy(person), time_seconds=time, appearance_gallery=gallery)
                used.add(j)
        for j, person in enumerate(fused):
            if j not in used:
                person["id"] = self.next_id
                self.next_id += 1
                self.tracks.append(dict(deepcopy(person), time_seconds=time, velocity=[0., 0.]))
            for detection in record["detections"]:
                if detection["detection_id"] in person["member_detection_ids"]:
                    detection["associated_id"] = person["id"]
        record["fused_people"] = fused
        record["headcounts"] = {r: sum(p["room_id"] == r for p in fused) for r in self.rooms}
        return record
