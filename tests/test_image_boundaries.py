import unittest
import numpy as np
from shapely.geometry import Polygon, Point
from src.boundaries import fit_boundary_room, refine_mount_pose, view_polygon


class ImageBoundaryTests(unittest.TestCase):
    def test_boundary_lines_ignore_extreme_outliers(self):
        rng = np.random.default_rng(5)
        observations = []
        for axis, side in [(0, -1), (0, 1), (1, -1), (1, 1)]:
            for value in np.linspace(-2.9, 2.9, 100):
                point = np.zeros(2)
                point[axis] = side * 3 + rng.normal(0, 0.006)
                point[1 - axis] = value
                observations.append({"camera": f"cam{axis}", "xy": point.tolist(), "uv": [0, 0]})
        for point in [[18, 18], [-18, 18], [18, -18], [-18, -18]]:
            observations.append({"camera": "outlier", "xy": point, "uv": [0, 0]})
        room, matrix, report = fit_boundary_room(observations, "r", [[-2.8, -2.8], [2.8, 2.8]])
        p = np.array(room["polygon"])
        np.testing.assert_allclose(p.min(0), [-3, -3], atol=0.04)
        np.testing.assert_allclose(p.max(0), [3, 3], atol=0.04)
        self.assertEqual(len(report["selected"]), 4)

    def test_pose_center_axis_and_ground_mapping_consistency(self):
        center = np.array([2.88, -2.5, 2.2])
        axis = -center / np.linalg.norm(center)
        right = np.cross(axis, [0, 0, 1])
        right /= np.linalg.norm(right)
        down = np.cross(axis, right)
        rotation = np.vstack([right, down, axis])
        translation = -rotation @ center
        k = np.array([[330, 0, 180], [0, 330, 144], [0, 0, 1.0]])
        h = np.linalg.inv(k @ np.column_stack([rotation[:, 0], rotation[:, 1], translation]))
        entry = {"H_image_to_map": h.tolist(), "K_estimated": k.tolist()}
        saved = np.array(entry["H_image_to_map"])
        pose = refine_mount_pose(entry, 360, 288, [[-3, -3], [3, -3], [3, 3], [-3, 3]])
        r = np.array(pose["R_world_to_camera"])
        t = np.array(pose["t_world_to_camera"])
        np.testing.assert_allclose(pose["camera_center_world"], -r.T @ t, atol=1e-10)
        np.testing.assert_allclose(pose["optical_axis_world"], r.T @ [0, 0, 1], atol=1e-10)
        np.testing.assert_array_equal(entry["H_image_to_map"], saved)
        self.assertLess(pose["wall_clearance_m"], 0.2)

    def test_fov_reaches_walls_and_is_not_fixed_length(self):
        polygon = [[-3, -3], [3, -3], [3, 3], [-3, 3]]
        center = np.array([2.88, -2.88])
        shape = Polygon(polygon)
        points = view_polygon(center, [-1, 1, -0.5], np.deg2rad(60), polygon)
        self.assertGreater(np.max(np.linalg.norm(points[1:] - center, axis=1)), 7)
        for point in points[1:]:
            self.assertLess(shape.boundary.distance(Point(point)), 1e-7)


if __name__ == "__main__":
    unittest.main()
