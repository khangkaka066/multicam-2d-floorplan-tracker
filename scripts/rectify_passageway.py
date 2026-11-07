#!/usr/bin/env python3
"""Compatibility wrapper: rectification now runs automatically during build-map."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_loader import RoomDataset
from src.map_builder import prepare_ground_rectification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/passageway"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/passageway"))
    args = parser.parse_args()
    prepare_ground_rectification(RoomDataset(args.data_dir), args.output_dir, force=True)


if __name__ == "__main__":
    main()
