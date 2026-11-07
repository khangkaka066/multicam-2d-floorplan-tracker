"""Manhattan wall fitting, explicit architectural elements and constrained room stitching."""

from copy import deepcopy
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
from .geometry import project


def room_from_polygon(room_id, polygon):
    p = np.asarray(polygon, float)
    shape = Polygon(p)
    if not shape.is_valid or shape.area <= 0:
        raise ValueError("Invalid room polygon")
    if not shape.exterior.is_ccw:
        p = p[::-1]
    walls = []
    for i, (a, b) in enumerate(zip(p, np.roll(p, -1, axis=0))):
        delta = b - a
        if min(abs(delta)) > 1e-7 or np.linalg.norm(delta) < 1e-6:
            raise ValueError("Room walls must be axis-aligned and nonzero")
        normal = np.array([delta[1], -delta[0]]) / np.linalg.norm(delta)
        walls.append(
            {
                "id": f"{room_id}_wall_{i}",
                "start": a.tolist(),
                "end": b.tolist(),
                "length_m": float(np.linalg.norm(delta)),
                "outward_normal": normal.tolist(),
                "thickness_m": 0.16,
                "openings": [],
            }
        )
    return {
        "id": room_id,
        "polygon": p.tolist(),
        "corners": p.tolist(),
        "walls": walls,
        "doors": [],
        "T_local_to_global": np.eye(3).tolist(),
    }


def vectorize_wall_segments(segments, snap_tolerance=0.15):
    """Close actual axis-aligned segment graphs, preserving L-shaped concavities."""
    lines = []
    points = np.asarray([p for segment in segments for p in segment], float)
    snapped = points.copy()
    for axis in range(2):
        remaining = list(range(len(points)))
        while remaining:
            seed = remaining.pop(0)
            group = [seed] + [
                i for i in remaining if abs(points[i, axis] - points[seed, axis]) <= snap_tolerance
            ]
            remaining = [i for i in remaining if i not in group]
            snapped[group, axis] = np.median(points[group, axis])
    for a, b in snapped.reshape(-1, 2, 2):
        if min(abs(b - a)) > 1e-6:
            raise ValueError("Non-Manhattan wall segment")
        lines.append(LineString([a, b]))
    polygons = list(polygonize(unary_union(lines)))
    if not polygons:
        raise ValueError("Wall observations do not form a closed boundary")
    polygon = max(polygons, key=lambda p: p.area).simplify(1e-8, preserve_topology=True)
    return np.asarray(polygon.exterior.coords[:-1])


def rigid_door_alignment(source_door, target_door, tolerance=0.10, gap_m=0.0):
    a, b = np.array(source_door["start"]), np.array(source_door["end"])
    c, d = np.array(target_door["end"]), np.array(target_door["start"])
    if abs(np.linalg.norm(b - a) - np.linalg.norm(d - c)) > tolerance:
        raise ValueError("Connected doorway widths disagree")
    angle = np.arctan2((d - c)[1], (d - c)[0]) - np.arctan2((b - a)[1], (b - a)[0])
    angle = round(angle / (np.pi / 2)) * (np.pi / 2)
    co, si = np.cos(angle), np.sin(angle)
    r = np.rint([[co, -si], [si, co]])
    if not np.isfinite(gap_m) or gap_m < 0:
        raise ValueError("gap_m must be finite and nonnegative")
    # target start->end follows the CCW room boundary: its right normal points outward.
    tangent = np.asarray(target_door["end"]) - target_door["start"]
    outward = np.array([tangent[1], -tangent[0]]) / np.linalg.norm(tangent)
    c, d = c + outward * gap_m, d + outward * gap_m
    t = (c + d) / 2 - r @ ((a + b) / 2)
    matrix = np.eye(3)
    matrix[:2, :2] = r
    matrix[:2, 2] = t
    if np.max(np.linalg.norm(project(matrix, [a, b]) - [c, d], axis=1)) > tolerance:
        raise ValueError("Doors cannot be snapped with an orthogonal rigid transform")
    return matrix


def transform_room(room, matrix):
    result = deepcopy(room)
    result["polygon"] = project(matrix, room["polygon"]).tolist()
    result["corners"] = deepcopy(result["polygon"])
    result["T_local_to_global"] = (
        np.asarray(matrix) @ room.get("T_local_to_global", np.eye(3))
    ).tolist()
    for wall in result["walls"]:
        wall["start"], wall["end"] = project(matrix, [wall["start"], wall["end"]]).tolist()
        wall["outward_normal"] = (np.asarray(matrix)[:2, :2] @ wall["outward_normal"]).tolist()
    for door in result["doors"]:
        door["start"], door["end"] = project(matrix, [door["start"], door["end"]]).tolist()
        if "center" in door:
            door["center"] = project(matrix, [door["center"]])[0].tolist()
        for event in door.get("events", []):
            if "xy" in event:
                event.setdefault("xy_scene", event["xy"])
                event["xy"] = project(matrix, [event["xy"]])[0].tolist()
    return result


def stitch_rooms(rooms, connections):
    """Place only connected rooms. Reject disconnected graphs, overlaps and inconsistent loops."""
    if not rooms:
        raise ValueError("No rooms")
    local = {r["id"]: deepcopy(r) for r in rooms}
    if len(local) != len(rooms):
        raise ValueError("Duplicate room IDs")
    placed = {rooms[0]["id"]: transform_room(rooms[0], np.eye(3))}
    pending = list(connections)
    while pending:
        progress = False
        for connection in pending[:]:
            a, b = connection["room_a"], connection["room_b"]
            if a == b:
                raise ValueError("Door connections must join different rooms")
            if a not in local or b not in local:
                raise ValueError("Unknown connected room")
            if a not in placed and b not in placed:
                continue
            if a not in placed:
                a, b = b, a
                da, db = connection["door_b"], connection["door_a"]
            else:
                da, db = connection["door_a"], connection["door_b"]
            target = next(d for d in placed[a]["doors"] if d["id"] == da)
            source = next(d for d in local[b]["doors"] if d["id"] == db)
            matrix = rigid_door_alignment(source, target, gap_m=connection.get("gap_m", 0.0))
            candidate = transform_room(local[b], matrix)
            if b in placed:
                if not np.allclose(placed[b]["T_local_to_global"], matrix, atol=0.1):
                    raise ValueError("Inconsistent room connection loop")
            else:
                for existing in placed.values():
                    if (
                        Polygon(candidate["polygon"])
                        .intersection(Polygon(existing["polygon"]))
                        .area
                        > 0.01
                    ):
                        raise ValueError("Snapped rooms overlap")
                placed[b] = candidate
            pending.remove(connection)
            progress = True
        if not progress:
            raise ValueError("Disconnected room graph; explicit doorway correspondences required")
    if len(placed) != len(rooms):
        raise ValueError("Unplaced rooms; no evidence of relative position")
    for connection in connections:
        for room_key, door_key, other_key in [
            ("room_a", "door_a", "room_b"),
            ("room_b", "door_b", "room_a"),
        ]:
            door = next(
                d for d in placed[connection[room_key]]["doors"] if d["id"] == connection[door_key]
            )
            door["connected_room"] = connection[other_key]
    shared = []
    values = list(placed.values())
    for i, a in enumerate(values):
        for b in values[i + 1 :]:
            for wa in a["walls"]:
                for wb in b["walls"]:
                    overlap = LineString([wa["start"], wa["end"]]).intersection(
                        LineString([wb["start"], wb["end"]])
                    )
                    if overlap.length > 0.1:
                        shared.append(
                            {
                                "rooms": [a["id"], b["id"]],
                                "walls": [wa["id"], wb["id"]],
                                "length_m": float(overlap.length),
                            }
                        )
    return values, {
        "connections": connections,
        "shared_walls": shared,
        "status": "aligned by rigid doorway constraints",
    }


def layout_rooms(rooms, topology=None, layout="stitched", atlas_gap_m=3.0):
    """Stitch evidenced components; arrange only unconnected components as an atlas."""
    from .topology import resolve_links

    if layout not in ["stitched", "atlas"]:
        raise ValueError("layout must be stitched or atlas")
    if not rooms or len({r["id"] for r in rooms}) != len(rooms):
        raise ValueError("Expected nonempty rooms with unique IDs")
    rooms = deepcopy(rooms)
    connections, matches = resolve_links(rooms, topology)
    for link in connections:
        if link.get("source") != "trajectory":
            continue
        openings = [
            next(
                d
                for r in rooms
                if r["id"] == link[room_key]
                for d in r["doors"]
                if d["id"] == link[door_key]
            )
            for room_key, door_key in [("room_a", "door_a"), ("room_b", "door_b")]
        ]
        widths = [np.linalg.norm(np.asarray(d["end"]) - d["start"]) for d in openings]
        if abs(widths[0] - widths[1]) <= 0.1 or all(d.get("confirmed") for d in openings):
            continue
        if abs(widths[0] - widths[1]) > 0.35:
            raise ValueError(
                "Inferred doorway widths disagree by more than 0.35 m; supply measured openings"
            )
        measured = [w for w, d in zip(widths, openings) if d.get("confirmed")]
        width = measured[0] if measured else min(widths)
        link["original_door_widths_m"] = [float(w) for w in widths]
        link["aligned_opening_width_m"] = float(width)
        for opening, old_width in zip(openings, widths):
            tangent = (np.asarray(opening["end"]) - opening["start"]) / old_width
            center = (np.asarray(opening["end"]) + opening["start"]) / 2
            opening["offset_m"] += (old_width - width) / 2
            opening.update(
                start=(center - tangent * width / 2).tolist(),
                end=(center + tangent * width / 2).tolist(),
                width_m=float(width),
            )
    chosen = connections if layout == "stitched" else []
    remaining = {r["id"] for r in rooms}
    components = []
    while remaining:
        seed = next(r["id"] for r in rooms if r["id"] in remaining)
        component = {seed}
        while True:
            expanded = component | {
                c[k]
                for c in chosen
                if {c["room_a"], c["room_b"]} & component
                for k in ["room_a", "room_b"]
            }
            if expanded == component:
                break
            component = expanded
        remaining -= component
        components.append(component)
    placed = []
    shared = []
    cursor = None
    for component in components:
        subset = [r for r in rooms if r["id"] in component]
        edges = [c for c in chosen if c["room_a"] in component]
        if edges:
            group, report = stitch_rooms(subset, edges)
            shared.extend(report["shared_walls"])
        else:
            group = [transform_room(subset[0], np.eye(3))]
        if cursor is not None:
            lower = np.min(np.vstack([r["polygon"] for r in group]), axis=0)[0]
            shift = np.eye(3)
            shift[0, 2] = cursor + atlas_gap_m - lower
            group = [transform_room(r, shift) for r in group]
        cursor = np.max(np.vstack([r["polygon"] for r in group]), axis=0)[0]
        placed.extend(group)
    placed = [next(p for p in placed if p["id"] == r["id"]) for r in rooms]
    complete = len(rooms) > 1 and len(components) == 1 and bool(chosen)
    return placed, {
        "status": "stitched" if complete else ("partial" if chosen else "atlas"),
        "requested_layout": layout,
        "connections": chosen,
        "shared_walls": shared,
        "components": [sorted(c) for c in components],
        "transition_matches": matches,
        "inferred_connection_count": sum(c.get("source") == "trajectory" for c in chosen),
        "physical_building_registration": complete,
        "registration_source": sorted({c.get("source", "config") for c in chosen}),
        "gap_is_physical_distance": bool(chosen),
        "display_gap_m": atlas_gap_m,
        "fallback_reason": (
            None
            if complete
            else "Unconnected components have no shared-clock transitions or declared adjacency; component spacing is display-only."
        ),
    }
