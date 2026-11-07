"""Perimeter camera symbols and wall-occluded FOVs, separate from calibrated poses."""
import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon, mapping
from shapely.ops import nearest_points


def perimeter_mount(center, direction, polygon, assigned_polygon=None):
    """Wall behind the pose; avoid internal seams when an assigned room is supplied."""
    center = np.asarray(center, float)[:2]
    axis = np.asarray(direction, float)[:2]
    norm = np.linalg.norm(axis)
    room = Polygon(polygon)
    if norm < 1e-9 or not room.is_valid or room.area <= 0:
        raise ValueError("A valid room and nonzero horizontal viewing direction are required")
    axis /= norm
    if not room.buffer(1e-8).covers(Point(center)):
        raise ValueError("Camera center lies outside its mounting envelope")
    if room.boundary.distance(Point(center)) < 1e-8:
        mount = center.copy()
    else:
        extent = np.linalg.norm(np.ptp(np.asarray(polygon), axis=0)) * 2 + 1
        hit = room.boundary.intersection(LineString([center, center - extent * axis]))
        points = []
        for part in getattr(hit, "geoms", [hit]):
            if part.geom_type == "Point":
                points.append(np.asarray(part.coords[0]))
            elif part.geom_type == "LineString":
                points.extend(np.asarray(part.coords))
        behind = [p for p in points if (center - p) @ axis >= -1e-8]
        if not behind:
            raise ValueError("Backward viewing ray does not meet the mounting perimeter")
        mount = min(behind, key=lambda p: np.linalg.norm(p - center))
    if assigned_polygon is not None:
        # HA/HA2 are contiguous floor components: their internal seam is not a mountable wall.
        # If the backward axis reaches that seam, choose the closest actual perimeter wall
        # in the backward half-plane while remaining inside the primary camera space.
        assigned_boundary = Polygon(assigned_polygon).boundary
        if assigned_boundary.distance(Point(mount)) > 1e-7:
            # Independently stitched coordinates may differ by a few floating-point ULPs.
            exterior = room.boundary.intersection(assigned_boundary.buffer(1e-8))
            candidates = []
            for segment in getattr(exterior, "geoms", [exterior]):
                if segment.geom_type not in ["LineString", "Point"]:
                    continue
                nearest = np.asarray(nearest_points(Point(center), segment)[1].coords[0])
                points = [nearest, *np.asarray(segment.coords)]
                for p in points:
                    if ((center - p) @ axis >= -1e-8 and
                            room.covers(Point(p + axis * 1e-6))):
                        candidates.append(p)
            if not candidates:
                raise ValueError("No actual perimeter wall behind the camera in its primary space")
            mount = min(candidates, key=lambda p: np.linalg.norm(p - center))
    if not room.covers(Point(mount + axis * 1e-6)):
        raise ValueError("Wall-mounted camera must face into its room")
    return mount


def wall_clipped_fov(mount, direction, published_visibility, polygon):
    """Cast rays to the first wall, then clip the published angular envelope to the room.

    The first-wall test also prevents light from crossing reentrant walls in L-shaped rooms;
    intersection alone would retain disconnected visible-looking patches behind those walls.
    """
    room = Polygon(polygon)
    mount = np.asarray(mount, float)
    axis = np.asarray(direction, float)[:2]
    axis /= np.linalg.norm(axis)
    origin = mount + axis * 1e-7  # numerical interior only; serialized mount stays on the wall
    if not room.covers(Point(origin)):
        raise ValueError("Visibility origin must face into the mounting envelope")
    vertices = np.asarray(room.exterior.coords[:-1])
    segments = np.roll(vertices, -1, axis=0) - vertices
    delta = vertices - origin
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    angles = np.unique(np.r_[angles - 1e-8, angles, angles + 1e-8,
                             np.linspace(-np.pi, np.pi, 181, endpoint=False)])
    angles = np.sort((angles + np.pi) % (2 * np.pi) - np.pi)
    hits = []
    for angle in angles:
        ray = np.array([np.cos(angle), np.sin(angle)])
        denominator = ray[0] * segments[:, 1] - ray[1] * segments[:, 0]
        nonparallel = abs(denominator) > 1e-12
        t = np.full(len(segments), np.inf)
        u = np.full(len(segments), np.inf)
        t[nonparallel] = ((delta[:, 0] * segments[:, 1] - delta[:, 1] * segments[:, 0]) /
                          np.where(nonparallel, denominator, 1))[nonparallel]
        u[nonparallel] = ((delta[:, 0] * ray[1] - delta[:, 1] * ray[0]) /
                          np.where(nonparallel, denominator, 1))[nonparallel]
        valid = (t >= 0) & (u >= -1e-10) & (u <= 1 + 1e-10)
        if not valid.any():
            raise ValueError("Visibility ray escaped a closed room boundary")
        hits.append(origin + ray * t[valid].min())
    visibility = Polygon(hits)
    if not visibility.is_valid:
        visibility = visibility.buffer(0)
    envelope = MultiPoint([mount.tolist(), *np.asarray(published_visibility).tolist()]).convex_hull
    clipped = visibility.intersection(envelope).intersection(room)
    if clipped.is_empty or clipped.area <= 1e-8 or clipped.geom_type not in ["Polygon", "MultiPolygon"]:
        raise ValueError("Camera has no polygonal floor visibility inside its room")
    return clipped


def attach_camera_layout(config, transforms, envelopes):
    """Populate render-only geometry; camera_xy_estimate/C_world/H remain calibrated."""
    rooms = {r["id"]: Polygon(r["polygon"]) for r in config["rooms"]}
    for name, entry in transforms["cameras"].items():
        envelope = envelopes[entry["room_id"]]
        polygon = envelope["polygon"]
        room_polygon = np.asarray(rooms[entry["room_id"]].exterior.coords[:-1]).tolist()
        mount = perimeter_mount(entry["camera_xy_estimate"], entry["optical_axis_world"], polygon,
                                assigned_polygon=room_polygon)
        fov = wall_clipped_fov(mount, entry["optical_axis_world"], entry["published_visibility_polygon"], polygon)
        fov = fov.intersection(rooms[entry["room_id"]])
        offset = np.asarray(entry["camera_xy_estimate"]) - mount
        axis = np.asarray(entry["optical_axis_world"])[:2]
        on_axis = abs(offset[0] * axis[1] - offset[1] * axis[0]) < 1e-7
        entry.update(render_mount_xy=mount.tolist(),
                     render_mount_offset_m=float(np.linalg.norm(mount - entry["camera_xy_estimate"])),
                     render_mount_method="backward_axis_intersection" if on_axis else "nearest_perimeter_behind_pose; internal seam avoided",
                     render_mount_status="inferred perimeter mount behind published pose on simplified room footprint; not a surveyed replacement for camera_center_world",
                     render_mount_envelope=polygon, render_mount_component=envelope["component"],
                     render_view_direction=np.asarray(entry["optical_axis_world"])[:2].tolist(),
                     render_fov_geometry=mapping(fov),
                     render_fov_method="published angular envelope, first-wall ray casting, assigned room intersection")
    config["camera_rendering"] = dict(
        mounts="render_mount_xy: backward wall snap; calibrated centers and projection matrices retained",
        fov="first solid wall; strictly within assigned room and primary camera space",
        hall_visibility="HA main hall only; HA2/HA3 connectivity extensions do not imply visibility")
