"""CITIC's published geometry and calibration in one canonical metric building frame."""
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import box, Polygon
from shapely.ops import unary_union

from .architecture import room_from_polygon, stitch_rooms, transform_room
from .geometry import project
from .map_builder import configure_canvas

ROOMS = {"passageway_pf": ["PF"], "passageway_pc": ["PC"],
         "showroom": ["SH"], "entrance_hall": ["HA", "HA2", "HA3"]}
LINKS = [("D01", "passageway_pf", "passageway_pc", [1.8, 6.3]),
         ("D02", "passageway_pc", "showroom", [10.64, 7.2]),
         ("D03", "passageway_pc", "entrance_hall", [21.0, 6.3])]


def add_door(room, name, center, width=0.9):
    center = np.asarray(center)
    for wall in room["walls"]:
        a, b = np.array(wall["start"]), np.array(wall["end"])
        tangent = (b - a) / wall["length_m"]
        offset = float((center - a) @ tangent)
        residual = np.linalg.norm(a + tangent * offset - center)
        if residual < 1e-7 and width / 2 <= offset <= wall["length_m"] - width / 2:
            door = dict(id=name, wall_id=wall["id"], offset_m=offset - width / 2,
                        width_m=width, start=(center - tangent * width / 2).tolist(),
                        end=(center + tangent * width / 2).tolist(), center=center.tolist(),
                        confirmed=True, source="published shared doorway center",
                        width_source="nominal 0.9m for drawing; dataset supplies centers only")
            room["doors"].append(door)
            wall["openings"].append(name)
            return
    raise ValueError(f"Door {name} does not lie on {room['id']} boundary")


class CiticDataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["dataset"] != "CITIC-MTMC":
            raise ValueError("Expected a CITIC-MTMC manifest")
        self.cameras = self.manifest["cameras"]
        self.metadata = self.root / self.manifest["metadata_dir"]
        self.nodes = {c["id"]: json.loads((self.metadata / "map_config" /
                     ("Node" + c["id"] + ".json")).read_text()) for c in self.cameras}

    def frame(self, camera, index, verification=False):
        cap = cv2.VideoCapture(str(self.root / camera["video"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"Cannot decode {camera['id']} source frame {index}")
        return frame

    def ground_truth(self):
        return np.loadtxt(self.root / self.manifest["ground_truth"], delimiter=",")


def build_calibration(dataset):
    """Stitch independent local polygons with shared doors, then center the camera network."""
    room_map = json.loads((dataset.metadata / "map_config/room_map.json").read_text())
    rectangles = {r["name"]: box(*np.array(r["rect"]).ravel())
                  for r in room_map["room_map"] if "name" in r}
    flip = np.diag([1., -1., 1.])
    rooms, source_to_local = [], {}
    for name, components in ROOMS.items():
        shape = unary_union([rectangles[c] for c in components]).simplify(1e-7)
        points = project(flip, np.asarray(shape.exterior.coords[:-1]))
        local = flip.copy()
        local[:2, 2] = -points.min(axis=0)
        room = room_from_polygon(name, project(local, np.asarray(shape.exterior.coords[:-1])))
        room.update(source_components=components, geometry_source="published room_map.json",
                    area_m2=shape.area)
        for door, a, b, center in LINKS:
            if name in [a, b]:
                # Verify shared center exists on both source components, including HA3.
                assert any(np.allclose(center, d) for c in components
                           for r in room_map["room_map"] if r.get("name") == c
                           for d in r["doors"])
                add_door(room, door, project(local, [center])[0])
        rooms.append(room)
        source_to_local[name] = local
    connections = [dict(room_a=a, room_b=b, door_a=d, door_b=d,
                        source="published shared doorway centers", gap_m=0.)
                   for d, a, b, _ in LINKS]
    placed, alignment = stitch_rooms(rooms, connections)
    source_to_map = np.asarray(placed[0]["T_local_to_global"]) @ source_to_local[placed[0]["id"]]
    for room in placed:
        np.testing.assert_allclose(np.asarray(room["T_local_to_global"]) @ source_to_local[room["id"]],
                                   source_to_map, atol=1e-8)
    centers = {}
    for name, node in dataset.nodes.items():
        cal = node["calibration"]
        if "R_t_mat" in cal:
            rt = np.asarray(cal["R_t_mat"])
            centers[name] = (-np.linalg.solve(rt[:, :3], rt[:, 3]))[:2]
        else:
            centers[name] = node["position"][:2]
    centroid = project(source_to_map, list(centers.values())).mean(axis=0)
    canonical = np.eye(3)
    canonical[:2, 2] = -centroid
    placed = [transform_room(r, canonical) for r in placed]
    source_to_map = canonical @ source_to_map
    entries = {}
    for camera in dataset.cameras:
        name = camera["id"]
        node = dataset.nodes[name]
        cal = node["calibration"]
        scale = np.diag([node["image"]["width"] / camera["width"],
                         node["image"]["height"] / camera["height"], 1.])
        if "homography" in cal:
            h_source = np.asarray(cal["homography"])
        else:
            rt = np.asarray(cal["R_t_mat"])
            k = np.asarray(cal["intr_mtx"])
            h_source = np.linalg.inv(k @ rt[:, [0, 1, 3]])
        h = source_to_map @ h_source @ scale
        h /= h[2, 2]
        xy = project(source_to_map, [centers[name]])[0]
        visibility = project(source_to_map, node["visibility_polygon"])
        aim = visibility[1:].mean(axis=0) - xy
        aim /= np.linalg.norm(aim)
        entry = dict(room_id=camera["room_id"], H_image_to_map=h.tolist(),
                     H_map_to_image=np.linalg.inv(h).tolist(), camera_xy_estimate=xy.tolist(),
                     declared_camera_position_source=node["position"],
                     published_visibility_polygon=visibility.tolist(),
                     optical_axis_world=[*aim, -0.4], calibration_image_size=[camera["width"], camera["height"]],
                     calibration_source=f"map_config/Node{name}.json",
                     camera_position_status="published position; homography-only camera",
                     extrinsics_available=False)
        if "R_t_mat" in cal:
            # Dataset z points down; canonical x/right, y/up, z/up is right handed.
            g3 = np.diag([1., -1., -1.])
            translation = np.r_[source_to_map[:2, 2], 0.]
            r = rt[:, :3] @ g3
            t = rt[:, 3] - r @ translation
            center = -np.linalg.solve(r, t)
            entry.update(K=k.tolist(), R_world_to_camera=r.tolist(), t_world_to_camera=t.tolist(),
                         camera_center_world=center.tolist(), optical_axis_world=r[2].tolist(),
                         camera_position_status="derived from published R_t_mat; declared position retained separately",
                         extrinsics_available=True)
        else:
            entry["camera_center_world"] = [*xy, node["position"][2]]
        entries[name] = entry
    alignment.update(status="stitched", method="rigid shared-door alignment of published room geometry",
                     trajectory_role="independent traversal verification; no GT used for geometry fit")
    config = dict(schema_version=4, dataset="CITIC-MTMC", sequence="capture1", units="meters",
                  source="published CITIC-MTMC room_map.json and Node*.json", rooms=placed,
                  physical_building_registration=True, alignment=alignment,
                  T_source_to_map=source_to_map.tolist(), coordinate_convention="right-handed; X right, Y up, Z up",
                  hall_components_note="HA + HA2 + HA3; HA3 is the real 3m vestibule linking hall to PC",
                  doorway_width_note="0.9m nominal illustration; only door centers are published",
                  geometry_estimated_from_video=False,
                  ground_truth_note="phone waypoint positions linearly interpolated by dataset authors",
                  minimum_display_clearance_m=0.)
    config.update(configure_canvas(placed, entries))
    transforms = dict(schema_version=4, dataset="CITIC-MTMC", units="meters",
                      convention="H_image_to_map maps source image pixels to canonical building meters",
                      gauge=dict(method="centroid of five camera ground positions", T_source_to_map=source_to_map.tolist()),
                      cameras=entries)
    for e in entries.values():
        e["H_image_to_canvas"] = (np.asarray(config["T_map_to_canvas"]) @ e["H_image_to_map"]).tolist()
    from .camera_layout import attach_camera_layout

    envelopes = {name: dict(component=components[0],
                           polygon=project(source_to_map, np.asarray(rectangles[components[0]].exterior.coords[:-1])).tolist())
                 for name, components in ROOMS.items()}
    attach_camera_layout(config, transforms, envelopes)
    return config, transforms
