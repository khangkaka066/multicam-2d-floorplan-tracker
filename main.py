#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import cv2
from src.dataset_loader import EPFLDataset
from src.calibrator import EPFLCalibrator
from src.map_builder import ArchitecturalMapBuilder
from src.geometry import save_json


def build(dataset, output):
    transforms = EPFLCalibrator().calibrate(dataset)
    image, texture, config = ArchitecturalMapBuilder(output).build(dataset, transforms)
    for name, img in [
        ("global_2d_map.png", image),
        ("global_map.png", image),
        (f"room_{dataset.manifest['room_id']}_map.png", image),
        ("structural_wall_support.png", texture),
    ]:
        if not cv2.imwrite(str(output / name), img):
            raise RuntimeError(f"Cannot write {name}")
    save_json(output / "transforms.json", transforms)
    save_json(output / "map_config.json", config)
    save_json(output / "dataset_profile.json", dataset.profile())
    from src.calibrator import floor_alignment_report

    save_json(output / "alignment_metrics.json", floor_alignment_report(dataset, transforms))
    print(
        "Architectural map/calibration built from depth wall planes; metric dimensions estimated.",
        flush=True,
    )


def run_projection(dataset, output, weights, frame_count, device, imgsz=None, show_raw=False):
    from src.observations import collect
    from src.projection_pipeline import process
    from src.bev_visualizer import export_verification
    from src.map_builder import draw_floorplan

    room_id = dataset.manifest["room_id"]
    scales = [384, imgsz or 960] if room_id == "laboratory" else [imgsz or 1280]
    collect(dataset.root, output, weights, scales, device, frame_count=frame_count)
    records = process(output)
    config = json.loads((output / "map_config.json").read_text())
    config["show_raw_projections"] = show_raw
    config["verification_record_index"] = min(78, len(records) - 1)
    save_json(output / "map_config.json", config)
    transforms = json.loads((output / "transforms_refined.json").read_text())
    image = draw_floorplan(config, transforms["cameras"])
    for name in ["global_2d_map.png", "global_map.png"]:
        if not cv2.imwrite(str(output / name), image):
            raise RuntimeError("Cannot save architectural map")
    export_verification(dataset, records, image, config, output)


def main():
    parser = argparse.ArgumentParser(description="Calibrated EPFL multi-camera BEV prototype")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "build-map", "calibrate", "project", "stitch", "build-global"],
        help="Default: reconstruct, calibrate, project both prepared EPFL scenes and render the atlas",
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: outputs/ for all/global; outputs/<room>/ for a single scene",
    )
    parser.add_argument(
        "--weights", type=Path, default=Path(__file__).resolve().parent / "models/yolo11n-pose.pt"
    )
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--imgsz",
        type=int,
        help="High-resolution pose pass (960 Laboratory / 1280 Passageway by default)",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Show faint raw observations and tethers behind fused people",
    )
    parser.add_argument(
        "--building",
        type=Path,
        help="Extracted room geometry and explicit door correspondences for stitch mode",
    )
    args = parser.parse_args()
    if not 1 <= args.frames <= 100:
        parser.error("--frames must be 1..100 for this sample")
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / "outputs"
        if args.mode in ["build-map", "calibrate", "project"]:
            manifest = args.data_dir / "manifest.json"
            if not manifest.exists():
                parser.error("Dataset missing: run python scripts/download_sample_data.py first")
            args.output_dir /= json.loads(manifest.read_text())["room_id"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    yolo_config = args.output_dir.resolve() / "yolo_config"
    yolo_config.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config))
    if args.mode == "all":
        from scripts.analyze_entries import main as analyze_entries
        from src.scene_atlas import build_global
        import torch

        torch.set_num_threads(2)
        for root in [args.data_dir, args.data_dir / "passageway"]:
            if not (root / "manifest.json").exists():
                parser.error(
                    "Prepare both scenes with the download scripts documented in README.md"
                )
        if not (args.data_dir / "passageway/metric_rectification.json").exists():
            parser.error("Run python scripts/rectify_passageway.py before the two-scene pipeline")
        for root in [args.data_dir, args.data_dir / "passageway"]:
            dataset = EPFLDataset(root)
            output = args.output_dir / dataset.manifest["room_id"]
            output.mkdir(parents=True, exist_ok=True)
            build(dataset, output)
            if dataset.manifest["room_id"] == "laboratory":
                analyze_entries(
                    ["--output-dir", str(output), "--data-dir", str(root), "--device", args.device]
                )
            run_projection(
                dataset, output, args.weights, args.frames, args.device, args.imgsz, args.show_raw
            )
        build_global(args.output_dir, args.data_dir)
        return
    if args.mode == "build-global":
        from src.scene_atlas import build_global

        build_global(args.output_dir, args.data_dir)
        return
    if args.mode == "stitch":
        if args.building is None:
            parser.error("--building is required for stitch")
        from src.architecture import stitch_rooms
        from src.map_builder import configure_canvas, draw_floorplan

        building = json.loads(args.building.read_text())
        rooms, alignment = stitch_rooms(building["rooms"], building["connections"])
        config = {
            "schema_version": 2,
            "units": "meters",
            "rooms": rooms,
            "alignment": alignment,
            "source": building.get("source", "explicit room/door constraints"),
        }
        config.update(configure_canvas(rooms))
        image = draw_floorplan(config)
        cv2.imwrite(str(args.output_dir / "global_2d_map.png"), image)
        save_json(args.output_dir / "map_config.json", config)
        print(json.dumps(alignment, indent=2))
        return
    dataset = EPFLDataset(args.data_dir)
    if args.mode in ["build-map", "calibrate"]:
        build(dataset, args.output_dir)
    else:
        if not (args.output_dir / "transforms.json").exists():
            parser.error("Run --mode build-map (or calibrate) for this scene before project")
        run_projection(
            dataset,
            args.output_dir,
            args.weights,
            args.frames,
            args.device,
            args.imgsz,
            args.show_raw,
        )


if __name__ == "__main__":
    main()
