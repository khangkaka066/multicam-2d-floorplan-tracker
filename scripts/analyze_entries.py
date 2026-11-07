#!/usr/bin/env python3
"""Discover doorway entry/exit zones for any manifest room using the core pipeline."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_loader import discover_rooms
from src.topology import analyze_scene


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--room")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args(argv)
    datasets = discover_rooms(args.data_dir)
    if args.room:
        datasets = [d for d in datasets if d.manifest["room_id"] == args.room]
    elif (args.output_dir / "map_config.json").exists():
        import json

        rooms = json.loads((args.output_dir / "map_config.json").read_text())["rooms"]
        if len(rooms) == 1:
            datasets = [d for d in datasets if d.manifest["room_id"] == rooms[0]["id"]]
    if not datasets:
        parser.error("No matching room")
    for dataset in datasets:
        output = args.output_dir
        local = output / dataset.manifest["room_id"]
        if local.exists():
            output = local
        analyze_scene(dataset, output, device=args.device, frame_count=args.frames)


if __name__ == "__main__":
    main()
