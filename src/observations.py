#!/usr/bin/env python3
"""Acquire independently cacheable high-resolution/multiscale pose observations."""
import argparse
import hashlib
import json
from pathlib import Path
from src.dataset_loader import EPFLDataset
from src.person_locator import MultiScalePoseLocator
from src.geometry import project, save_json


def observation_metadata(dataset, weights, scales, frame_count):
    return {
        "weights_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
        "scales": scales,
        "confidence": 0.25,
        "appearance_model": "resnet18",
        "appearance_weights_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "models/resnet18-f37072fd.pth").read_bytes()
        ).hexdigest(),
        "source_frame_hashes": [
            dataset.observation_fingerprint(c, frame_count) for c in dataset.cameras
        ],
        "frames": frame_count,
        "source_starts": [c["verification_start_frame"] for c in dataset.cameras],
        "room_id": dataset.manifest["room_id"],
        "cameras": [c["id"] for c in dataset.cameras],
    }


def collect(data_dir, output, weights, scales, device="cpu", frame_count=100):
    import torch

    torch.set_num_threads(2)
    dataset = data_dir if isinstance(data_dir, EPFLDataset) else EPFLDataset(data_dir)
    output = Path(output)
    transforms = json.loads((output / "transforms.json").read_text())
    metadata = observation_metadata(dataset, weights, scales, frame_count)
    path = output / "pose_observations.json"
    meta = output / "pose_observations_source.json"
    if path.exists() and meta.exists() and json.loads(meta.read_text()) == metadata:
        return json.loads(path.read_text())
    locator = MultiScalePoseLocator(weights, scales, device=device)
    records = []
    for index in range(frame_count):
        record = {
            "index": index,
            "source_frame": dataset.cameras[0]["verification_start_frame"] + index,
            "time_seconds": (dataset.cameras[0]["verification_start_frame"] + index)
            / dataset.cameras[0]["fps"],
            "detections": [],
        }
        for camera in dataset.cameras:
            for ordinal, d in enumerate(
                locator.detect(dataset.frame(camera, index, verification=True))
            ):
                d.update(
                    camera=camera["id"],
                    room_id=camera["room_id"],
                    detection_id=f'{index}:{camera["id"]}:{ordinal}',
                )
                d["xy"] = project(transforms["cameras"][camera["id"]]["H_image_to_map"], [d["uv"]])[
                    0
                ].tolist()
                record["detections"].append(d)
        records.append(record)
        if index % 10 == 0:
            print(
                dataset.manifest["room_id"],
                "multiscale pose",
                index + 1,
                "count",
                len(record["detections"]),
                flush=True,
            )
    save_json(path, records)
    save_json(meta, metadata)
    return records


def main():
    from .dataset_loader import discover_rooms

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--room")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    for dataset in discover_rooms(args.data_dir):
        if args.room and dataset.manifest["room_id"] != args.room:
            continue
        collect(
            dataset,
            args.output_dir / dataset.manifest["room_id"],
            Path("models/yolo11n-pose.pt"),
            dataset.manifest.get("pose_scales", [1280]),
            args.device,
            args.frames,
        )


if __name__ == "__main__":
    main()
