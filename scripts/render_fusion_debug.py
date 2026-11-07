#!/usr/bin/env python3
"""Render optional faint raw-camera tethers behind the actual fused people."""
import argparse
import json
from pathlib import Path
import sys
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scene_atlas import RealSceneDataset
from src.bev_visualizer import PanelVisualizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=78)
    args = parser.parse_args()
    root = Path("outputs")
    config = json.loads((root / "map_config.json").read_text())
    config["show_raw_projections"] = True
    record = next(
        r for r in json.loads((root / "projections.json").read_text()) if r["index"] == args.index
    )
    base = cv2.imread(str(root / "global_2d_map.png"))
    dataset = RealSceneDataset([("LAB", "data"), ("PASS", "data/passageway")])
    image = PanelVisualizer().render(dataset, record, base, config)
    left = image.shape[1] - base.shape[1]
    cv2.putText(
        image,
        "Faint dots/tethers: raw observations; rejected points do not affect headcount",
        (left + 35, 110),
        0,
        0.40,
        (100, 90, 75),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(root / "verification_bev_debug.png"), image)
    cv2.imwrite(str(root / "fused_bev_debug.png"), image[: base.shape[0], left:])


if __name__ == "__main__":
    main()
