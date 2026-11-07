#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import cv2
from src.map_builder import ArchitecturalMapBuilder, prepare_ground_rectification
from src.geometry import save_json


def build(dataset, output):
    transforms = prepare_ground_rectification(dataset, output)
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

    scales = dataset.manifest.get("pose_scales", [1280])
    if imgsz:
        scales = list(dict.fromkeys([*scales[:-1], imgsz]))
    collect(dataset, output, weights, scales, device, frame_count=frame_count)
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
    from src.dataset_loader import discover_rooms
    from src.scene_atlas import assemble
    from src.topology import analyze_scene

    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Manifest-based multi-room floorplans, topology and BEV fusion"
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "build-map", "calibrate", "project", "stitch", "build-global"],
    )
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--room", help="Process only this room ID")
    parser.add_argument(
        "--layout",
        choices=["stitched", "atlas"],
        default="stitched",
        help="Default: transition/config doorway snapping, then atlas fallback",
    )
    parser.add_argument("--topology", type=Path, help="Default: configs/topology.json if present")
    parser.add_argument("--weights", type=Path, default=root / "models/yolo11n-pose.pt")
    parser.add_argument("--frames", type=int, help="Verification samples; default CITIC 360, EPFL 100")
    parser.add_argument("--sample-stride", type=int, default=12,
                        help="CITIC: source frames between synchronized samples (25 source FPS)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--show-raw", action="store_true")
    parser.add_argument(
        "--building", type=Path, help="Legacy explicit room/door geometry for stitch mode"
    )
    args = parser.parse_args()
    if (args.frames is not None and args.frames < 1) or args.sample_stride < 1:
        parser.error("--frames and --sample-stride must be positive")
    citic_root = root / "data/citic_mtmc"
    # Prefer the real connected-building benchmark when prepared; explicit EPFL paths still work.
    if (args.data_dir == root / "data" and (citic_root / "manifest.json").exists()
            and args.room is None and args.mode != "stitch"):
        args.data_dir = citic_root
    manifest_path = args.data_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("dataset") == "CITIC-MTMC":
        if args.mode == "stitch" or args.layout == "atlas" or args.building or args.room:
            parser.error("CITIC uses its published connected building; choose all/build-map/calibrate/project/build-global")
        from src.citic_pipeline import run
        args.frames = args.frames or 360
        run(args, root)
        return
    args.frames = args.frames or 100
    if args.topology is None and (root / "configs/topology.json").exists():
        args.topology = root / "configs/topology.json"
    if args.mode == "stitch":
        if args.building is None:
            parser.error("--building is required for stitch")
        from src.architecture import stitch_rooms
        from src.map_builder import configure_canvas, draw_floorplan

        building = json.loads(args.building.read_text())
        rooms, alignment = stitch_rooms(building["rooms"], building["connections"])
        config = dict(
            schema_version=2,
            units="meters",
            rooms=rooms,
            alignment=alignment,
            source=building.get("source", "explicit room/door constraints"),
        )
        config.update(configure_canvas(rooms))
        output = args.output_dir or root / "outputs"
        output.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output / "global_2d_map.png"), draw_floorplan(config))
        save_json(output / "map_config.json", config)
        return
    datasets = discover_rooms(args.data_dir)
    if args.room:
        datasets = [d for d in datasets if d.manifest["room_id"] == args.room]
    if not datasets:
        parser.error("No matching room manifest")
    output = args.output_dir or root / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output.resolve() / "yolo_config"))
    if args.mode == "build-global":
        assemble(output, datasets, args.layout, args.topology)
        return
    import torch

    torch.set_num_threads(2)
    for dataset in datasets:
        name = dataset.manifest["room_id"]
        local = output / name if len(datasets) > 1 or args.output_dir is None else output
        local.mkdir(parents=True, exist_ok=True)
        frame_count = min(
            args.frames, min(c.get("verification_frames", c["frames"]) for c in dataset.cameras)
        )
        if args.mode in ["all", "build-map", "calibrate"]:
            build(dataset, local)
        if args.mode in ["all", "project"]:
            if not (local / "transforms.json").exists():
                parser.error(f"Build room {name} before projecting")
            run_projection(
                dataset, local, args.weights, frame_count, args.device, args.imgsz, args.show_raw
            )
        analyze_scene(dataset, local, args.weights, args.device, frame_count)
        from src.map_builder import draw_floorplan

        config = json.loads((local / "map_config.json").read_text())
        filename = (
            "transforms_refined.json" if args.mode in ["all", "project"] else "transforms.json"
        )
        transforms = json.loads((local / filename).read_text())
        local_image = draw_floorplan(config, transforms["cameras"])
        cv2.imwrite(str(local / "global_2d_map.png"), local_image)
        if len(datasets) == 1 and args.mode in ["all", "project"]:
            from src.bev_visualizer import export_verification

            export_verification(
                dataset,
                json.loads((local / "projections.json").read_text()),
                local_image,
                config,
                local,
            )
    if len(datasets) > 1:
        assemble(
            output,
            datasets,
            args.layout,
            args.topology,
            maps_only=args.mode in ["build-map", "calibrate"],
        )


if __name__ == "__main__":
    main()
