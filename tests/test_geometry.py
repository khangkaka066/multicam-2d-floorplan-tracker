import unittest
from pathlib import Path
import numpy as np
from src.geometry import project
from src.person_locator import ground_point
from src.calibrator import EPFLCalibrator
from src.dataset_loader import EPFLDataset


class GeometryTests(unittest.TestCase):
    @unittest.skipUnless(
        Path("data/manifest.json").exists(), "Download EPFL samples for this integration test"
    )
    def test_real_calibration_resizing_and_roundtrip(self):
        data = EPFLDataset("data")
        a = EPFLCalibrator().calibrate(data)
        for c in data.cameras:
            c["width"] *= 2
            c["height"] *= 2
        b = EPFLCalibrator().calibrate(data)
        # Canonical origins vary slightly with the approximate camera VP; compare source coordinates.
        for camera in a["cameras"]:
            points = np.array([[60, 160], [180, 250], [280, 200]])
            p = project(a["cameras"][camera]["H_image_to_map"], points) + a["gauge"]["origin_epfl"]
            q = (
                project(b["cameras"][camera]["H_image_to_map"], points * 2)
                + b["gauge"]["origin_epfl"]
            )
            np.testing.assert_allclose(p, q, atol=1e-8)
            recovered = project(
                a["cameras"][camera]["H_map_to_image"],
                project(a["cameras"][camera]["H_image_to_map"], points),
            )
            np.testing.assert_allclose(recovered, points, atol=1e-8)

    def test_horizon_rejected(self):
        with self.assertRaises(ValueError):
            project([[1, 0, 0], [0, 1, 0], [0, 1, -10]], [[3, 10]])

    def test_occlusion_levels_and_missing_pose(self):
        k = np.zeros((17, 3))
        self.assertIsNone(ground_point(k, [0, 0, 100, 200]))
        k[0] = [50, 20, 0.9]
        for ids, expected, method in [
            ([5, 6], 120, "shoulders_x5"),
            ([7, 8], 140, "elbows_x3"),
            ([11, 12], 180, "hips_x2"),
            ([13, 14], 212, "knees_x1.6"),
        ]:
            height = {5: 40, 7: 60, 11: 100, 13: 140}[ids[0]]
            for i in ids:
                k[i] = [50, height, 0.9]
            result = ground_point(k, [0, 0, 100, 200])
            self.assertEqual(result[1], method)
            self.assertAlmostEqual(result[0][1], expected)
        k[15] = [45, 195, 0.9]
        k[16] = [55, 195, 0.9]
        np.testing.assert_allclose(ground_point(k, [0, 0, 100, 200])[0], [50, 199])


if __name__ == "__main__":
    unittest.main()
