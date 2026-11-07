import json
from pathlib import Path
import numpy as np


def project(matrix, points):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-9):
        raise ValueError("Projection reaches the projective horizon")
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
