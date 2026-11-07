"""Mount/FOV regressions: calibrated poses stay intact while display geometry respects walls."""
from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon, shape

from src.camera_layout import perimeter_mount, wall_clipped_fov
from src.citic_dataset import CiticDataset, build_calibration
from src.citic_visualizer import camera_fov_mask, camera_label_layout, _overlap
from src.geometry import project


def test_mount_hits_first_wall_behind_view_instead_of_closest_side_wall():
    room = [[0, 0], [2, 0], [2, 12], [0, 12]]
    mount = perimeter_mount([.2, 2], [0, 1], room)
    np.testing.assert_allclose(mount, [.2, 0])
    assert Polygon(room).boundary.distance(Point(mount)) == 0


def test_mount_preserves_an_existing_wall_position_and_rejects_outward_axis():
    room = [[0, 0], [4, 0], [4, 4], [0, 4]]
    np.testing.assert_allclose(perimeter_mount([0, 2], [1, 0], room), [0, 2])
    with pytest.raises(ValueError, match="face into"):
        perimeter_mount([0, 2], [-1, 0], room)


def test_internal_component_seam_is_not_mistaken_for_an_external_mount_wall():
    primary = [[0, 0], [4, 0], [4, 4], [0, 4]]
    hall_with_extension = [[0, 0], [4, 0], [4, 6], [2, 6], [2, 4], [0, 4]]
    mount = perimeter_mount([3, 3], [-.2, -1], primary, hall_with_extension)
    np.testing.assert_allclose(mount, [4, 3])


def test_first_wall_occludes_concave_room_arm_that_intersection_alone_would_retain():
    room = [[0, 0], [6, 0], [6, 2], [2, 2], [2, 6], [0, 6]]
    visibility = wall_clipped_fov([6, 1], [-1, 0], room, room)
    assert visibility.difference(Polygon(room)).area < 1e-10
    assert visibility.covers(Point(1, 1))
    assert not visibility.covers(Point(1, 5))
    for point in np.asarray(visibility.exterior.coords):
        assert Polygon(room).buffer(1e-7).covers(LineString([[6, 1], point]))


@pytest.fixture
def citic_layout():
    if not Path("data/citic_mtmc/manifest.json").exists():
        pytest.skip("Download CITIC metadata for camera layout integration")
    return build_calibration(CiticDataset("data/citic_mtmc"))


def test_all_five_mounts_are_on_their_primary_perimeters_and_behind_calibrated_centers(citic_layout):
    config, transforms = citic_layout
    source = np.linalg.inv(config["T_source_to_map"])
    for name, entry in transforms["cameras"].items():
        mount, center = np.array(entry["render_mount_xy"]), np.array(entry["camera_xy_estimate"])
        axis = np.asarray(entry["optical_axis_world"])[:2]
        axis /= np.linalg.norm(axis)
        perimeter = Polygon(entry["render_mount_envelope"])
        assert perimeter.boundary.distance(Point(mount)) < 1e-7
        assert (center - mount) @ axis >= -1e-7
        room = next(r for r in config["rooms"] if r["id"] == entry["room_id"])
        assert Polygon(room["polygon"]).boundary.distance(Point(mount)) < 1e-7
        if name != "HallSeg":
            np.testing.assert_allclose(center - mount, np.linalg.norm(center - mount) * axis, atol=1e-7)
        assert "inferred" in entry["render_mount_status"]
        assert np.linalg.norm(mount - center) > .1
        source_mount = project(source, [mount])[0]
        if name == "PF":
            assert source_mount[1] == pytest.approx(13.2)
        elif name == "PC":
            assert source_mount[0] == pytest.approx(1.8)
        elif name == "Showroom":
            assert source_mount[0] == pytest.approx(1.8)
        else:
            assert entry["render_mount_component"] == "HA"
            if name == "HallSeg":
                assert source_mount[0] == pytest.approx(30.35)
        if entry["extrinsics_available"]:
            r, t = np.asarray(entry["R_world_to_camera"]), np.asarray(entry["t_world_to_camera"])
            np.testing.assert_allclose(center, (-np.linalg.solve(r, t))[:2], atol=1e-8)


def test_fov_geometry_and_raster_never_bleed_into_adjacent_rooms(citic_layout):
    config, transforms = citic_layout
    width, height = config["canvas_size"]
    rooms = {r["id"]: Polygon(r["polygon"]) for r in config["rooms"]}
    for entry in transforms["cameras"].values():
        geometry = shape(entry["render_fov_geometry"])
        assert geometry.is_valid and geometry.area > 0
        assert geometry.difference(rooms[entry["room_id"]]).area < 1e-10
        assert geometry.difference(Polygon(entry["render_mount_envelope"])).area < 1e-10
        for name, room in rooms.items():
            if name != entry["room_id"]:
                assert geometry.intersection(room).area < 1e-10
        room = next(r for r in config["rooms"] if r["id"] == entry["room_id"])
        room_mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(room_mask, [np.int32(np.rint(project(config["T_map_to_canvas"], room["polygon"])))], 255)
        fill, outline = camera_fov_mask(config, entry)
        assert not np.any((fill > 0) & (room_mask == 0))
        assert not np.any((outline > 0) & (room_mask == 0))


def test_camera_names_avoid_people_doors_and_other_camera_names(citic_layout):
    config, transforms = citic_layout
    # Put a person at an initial preferred callout to force placement to adapt.
    initial = camera_label_layout(config, transforms)
    rect = initial["Showroom"]["rect"]
    pixel = [[(rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2]]
    xy = project(np.linalg.inv(config["T_map_to_canvas"]), pixel)[0]
    record = dict(fused_people=[dict(id=12, xy=xy.tolist())])
    layouts = camera_label_layout(config, transforms, record, [(12, xy.tolist())])
    point = pixel[0]
    obstacle = (point[0] - 15, point[1] - 28, point[0] + 65, point[1] + 30)
    rectangles = [v["rect"] for v in layouts.values()]
    assert all(not _overlap(r, obstacle) for r in rectangles)
    assert all(not _overlap(a, b) for i, a in enumerate(rectangles) for b in rectangles[i + 1:])
    for room in config["rooms"]:
        boundary = np.zeros((config["canvas_size"][1], config["canvas_size"][0]), np.uint8)
        cv2.polylines(boundary, [np.int32(np.rint(project(config["T_map_to_canvas"], room["polygon"])))], True, 1, 4)
        for x1, y1, x2, y2 in rectangles:
            assert not boundary[y1:y2, x1:x2].any()
        for door in room["doors"]:
            p = project(config["T_map_to_canvas"], [door["center"]])[0]
            obstacle = (p[0] - 8, p[1] - 25, p[0] + 45, p[1] + 9)
            assert all(not _overlap(r, obstacle) for r in rectangles)
