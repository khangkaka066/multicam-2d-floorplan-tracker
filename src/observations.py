#!/usr/bin/env python3
"""Acquire independently cacheable high-resolution/multiscale pose observations."""
import argparse
import hashlib
import json
from pathlib import Path
from src.dataset_loader import EPFLDataset
from src.person_locator import MultiScalePoseLocator
from src.geometry import project, save_json


def collect(data_dir, output, weights, scales, device="cpu", frame_count=100):
    import torch

    torch.set_num_threads(2)
    dataset = EPFLDataset(data_dir)
    output = Path(output)
    transforms = json.loads((output / "transforms.json").read_text())
    metadata = {
        "weights_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
        "scales": scales,
        "confidence": 0.25,
        "appearance_model": "resnet18",
        "frames": frame_count,
        "source_starts": [c["verification_start_frame"] for c in dataset.cameras],
    }
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
    p = argparse.ArgumentParser()
    p.add_argument("--scene", choices=["laboratory", "passageway"], required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    collect(
        "data" if a.scene == "laboratory" else "data/passageway",
        f"outputs/{a.scene}",
        "models/yolo11n-pose.pt",
        [384, 960] if a.scene == "laboratory" else [1280],
        a.device,
    )


if __name__ == "__main__":
    main()
