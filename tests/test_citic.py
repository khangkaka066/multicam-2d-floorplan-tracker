"""Regression checks for connected-building coordinate and observation contracts."""
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from src.architecture import room_from_polygon
from src.building_tracker import BuildingTracker, associate_building_frame
from src.citic_evaluation import gt_at_frame
from src.citic_optimization import refine_overlap
from src.geometry import project
from scripts.download_citic_mtmc import zip_index


def observation(camera, xy, feature=(1., 0.)):
    return dict(camera=camera, xy=list(xy), uv=list(xy), appearance=list(feature),
                display_accepted=True, detection_id=camera + str(xy), method="ankles",
                confidence=0.9, ankle_confidence=0.9)


def test_disjoint_camera_observations_are_not_discarded():
    detections = [observation("hall", [0, 0]), observation("pf", [10, 0]),
                  observation("pc", [5, 0]), observation("showroom", [5, 10])]
    groups = associate_building_frame(detections)
    assert len(groups) == 4
    assert sorted(d["detection_id"] for group in groups for d in group) == sorted(d["detection_id"] for d in detections)


def test_overlap_fuses_once_per_camera_and_preserves_new_people():
    detections = [observation("hall_a", [0, 0]), observation("hall_b", [0.1, 0]),
                  observation("hall_b", [2, 0]), observation("pf", [10, 0])]
    groups = associate_building_frame(detections)
    assert sorted(map(len, groups)) == [1, 1, 2]
    assert all(len({d["camera"] for d in g}) == len(g) for g in groups)


def test_close_cross_view_contacts_tolerate_front_back_appearance_change():
    detections = [observation("wide", [1., 2.]),
                  observation("reverse", [1.1, 2.1], feature=(.5, .866))]
    assert len(associate_building_frame(detections)) == 1


def test_gt_frame_index_is_one_based_and_does_not_scale_meters():
    gt = np.array([[1, 8, -1, -1, -1, -1, -1, 2., 3., 0.],
                   [2, 9, -1, -1, -1, -1, -1, 4., 5., 0.]])
    matrix = np.array([[1., 0., -10.], [0., -1., 7.], [0., 0., 1.]])
    result = gt_at_frame(gt, 0, matrix)
    assert result == [(8, [-8., 4.])]
    assert gt_at_frame(gt, 5, matrix) == []


def test_tracking_accepts_one_camera_and_handoffs_through_shared_door():
    a = room_from_polygon("a", [[0, 0], [2, 0], [2, 2], [0, 2]])
    b = room_from_polygon("b", [[2, 0], [4, 0], [4, 2], [2, 2]])
    a["doors"] = [dict(id="D01", center=[2, 1])]
    config = dict(rooms=[a, b], alignment=dict(connections=[dict(room_a="a", room_b="b", door_a="D01")]))
    cameras = {name: dict(H_image_to_map=np.eye(3).tolist(), camera_xy_estimate=[0, 0]) for name in ["left", "right"]}
    tracker = BuildingTracker(config, dict(cameras=cameras))
    first = dict(time_seconds=0., source_frame=0, detections=[observation("left", [1.8, 1])])
    second = dict(time_seconds=1., source_frame=25, detections=[observation("right", [2.3, 1])])
    tracker.update(first)
    tracker.update(second)
    assert first["fused_people"][0]["id"] == second["fused_people"][0]["id"] == 1
    assert tracker.transitions[0]["from_room"] == "a"
    assert tracker.transitions[0]["to_room"] == "b"

    # Real corridor blind intervals can outlast a short single-view tracker timeout.
    third = dict(time_seconds=22., source_frame=550, detections=[observation("left", [1.9, 1])])
    tracker.update(third)
    assert third["fused_people"][0]["id"] == 1


def test_overlap_refinement_rejects_validation_regression():
    records = []
    for index in range(30):
        residual = .2 if index < 15 else -.2
        a = observation("HallWide", [residual, 0])
        b = observation("HallSeg", [0, 0])
        for d in [a, b]:
            d["foreground_supported"] = True
        records.append(dict(source_frame=index, detections=[a, b]))
    transforms = dict(cameras={"HallSeg": dict(H_image_to_map=np.eye(3).tolist())})
    result, report = refine_overlap(records, transforms)
    assert not report["accepted"]
    assert result == transforms
    assert report["heldout_mean_after_m"] > report["heldout_mean_before_m"]


@pytest.mark.skipif(not Path("data/citic_mtmc/manifest.json").exists(), reason="Download public CITIC capture1")
def test_published_citic_stitch_and_calibration_are_consistent():
    from src.citic_dataset import CiticDataset, build_calibration
    config, transforms = build_calibration(CiticDataset("data/citic_mtmc"))
    assert len(config["rooms"]) == 4 and len(transforms["cameras"]) == 5
    assert unary_union([Polygon(r["polygon"]) for r in config["rooms"]]).geom_type == "Polygon"
    assert config["physical_building_registration"]
    hall = next(r for r in config["rooms"] if r["id"] == "entrance_hall")
    assert hall["source_components"] == ["HA", "HA2", "HA3"]
    np.testing.assert_allclose(np.mean([c["camera_xy_estimate"] for c in transforms["cameras"].values()], axis=0), [0, 0], atol=1e-8)
    for e in transforms["cameras"].values():
        h, hi = np.array(e["H_image_to_map"]), np.array(e["H_map_to_image"])
        np.testing.assert_allclose(h @ hi, np.eye(3), atol=1e-8)
        if e["extrinsics_available"]:
            r, t, k = np.array(e["R_world_to_camera"]), np.array(e["t_world_to_camera"]), np.array(e["K"])
            hp = k @ np.column_stack([r[:, :2], t])
            product = hp @ h
            np.testing.assert_allclose(product / product[2, 2], np.eye(3), atol=1e-8)
            assert e["camera_center_world"][2] > 0


def test_zip_index_uses_true_offsets_and_rejects_missing_eocd():
    import io
    import zipfile
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("capture1/PC.mp4", b"test video" * 10)
    data = stream.getvalue()
    result = zip_index(data, len(data))
    assert result[0]["name"] == "capture1/PC.mp4"
    assert result[0]["offset"] == 0 and result[0]["size"] == 100
    with pytest.raises(ValueError, match="end record"):
        zip_index(b"not a zip", 9)


def test_large_extrapolation_requires_strong_pose_evidence():
    from src.citic_foreground import pose_supported
    keypoints = np.zeros((17, 3))
    keypoints[:3, 2] = .9
    weak = dict(method="shoulders_x5", detector_confidence=.4, keypoints=keypoints.tolist())
    assert not pose_supported(weak)
    assert pose_supported(dict(weak, detector_confidence=.9))
    assert pose_supported(dict(weak, method="ankles"))


def test_reflection_contact_is_rejected_without_erasing_ambiguous_bright_floor():
    from src.citic_foreground import FloorSupport
    plugin = FloorSupport.__new__(FloorSupport)
    camera = dict(id="view", width=10, height=10)
    for label, expected in [(3, True), (10, False), (19, False), (0, True)]:
        plugin.labels = {"view": np.full((10, 10), label, np.uint8)}
        detection = dict(uv=[5., 5.])
        plugin.score(detection, camera)
        assert detection["floor_supported"] is expected
    detection = dict(uv=[5., 15.])
    plugin.score(detection, camera)
    assert detection["floor_supported"] and detection["floor_support_fraction"] is None


def test_doorway_event_comparison_preserves_misses_and_wrong_directions():
    from src.citic_evaluation import evaluate_transitions
    predicted = [dict(id=1, from_room="a", to_room="b", source_frame=100),
                 dict(id=1, from_room="b", to_room="a", source_frame=120)]
    truth = [dict(gt_id=9, from_room="a", to_room="b", source_frame=110),
             dict(gt_id=9, from_room="b", to_room="c", source_frame=120)]
    result = evaluate_transitions(predicted, truth)
    assert result["matched_events"] == 1
    assert result["precision"] == result["recall"] == .5
