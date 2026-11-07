#!/usr/bin/env python3
"""Fetch real EPFL Passageway clips plus a longer Laboratory interval for entry/exit analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_sample_data import fetch, extract, PAGE, BASE


def timeline(camera_folder, url, start=0, duration=150, fps=2):
    target = camera_folder / "timeline.avi"
    if not target.exists():
        temporary = camera_folder / "timeline.part.avi"
        with tempfile.TemporaryDirectory(prefix="epfl-timeline-") as temporary_folder:
            prefix = Path(temporary_folder) / "prefix.avi"
            fetch(url, prefix, 32 * 1024**2)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(prefix),
                    "-t",
                    "45",
                    "-vf",
                    f"fps={fps}",
                    "-an",
                    "-c:v",
                    "mjpeg",
                    str(temporary),
                ],
                check=True,
                timeout=90,
            )
        temporary.replace(target)

    return {
        "video": str(target),
        "sampling_fps": fps,
        "source_start_seconds": start,
        "requested_duration_seconds": duration,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--skip-lab-timeline", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    scene = root / "passageway"
    scene.mkdir(parents=True, exist_ok=True)
    calibration = scene / "calibration-passageway.txt"
    fetch(
        "https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/calibration-passageway.txt",
        calibration,
    )
    manifest = {
        "dataset": "EPFL Passageway",
        "sequence": "passageway1",
        "source": PAGE,
        "license": "CVLab EPFL copyright; research use",
        "room_id": "passageway",
        "calibration": calibration.name,
        "calibration_image_size": [720, 576],
        "coordinate_units": "EPFL passageway top-view units",
        "synchronized": True,
        "first_source_frame": 0,
        "physical_connection_to_laboratory": None,
        "cameras": [],
    }
    for i in range(4):
        folder = scene / f"cam{i}"
        frames = folder / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        server = "video3" if i == 3 else "video2"
        url = f"https://documents.epfl.ch/groups/c/cv/cvlab-pom-{server}/www/passageway1-c{i}.avi"
        video = folder / "sample.avi"
        if not video.exists():
            with tempfile.TemporaryDirectory(prefix="epfl-passageway-") as temp:
                prefix = Path(temp) / "prefix.avi"
                fetch(url, prefix, 8 * 1024**2)
                info = extract(prefix, video, frames, 100)
        else:
            cap = cv2.VideoCapture(str(video))
            info = {
                "width": int(cap.get(3)),
                "height": int(cap.get(4)),
                "fps": cap.get(5),
                "frames": int(cap.get(7)),
            }
            cap.release()
        info.update(
            {
                "id": f"cam{i}",
                "room_id": "passageway",
                "source_url": url,
                "video": str(video.relative_to(scene)),
                "frames_dir": str(frames.relative_to(scene)),
                "verification_video": str(video.relative_to(scene)),
                "verification_frames_dir": str(frames.relative_to(scene)),
                "verification_start_frame": 0,
                "sample_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            }
        )
        manifest["cameras"].append(info)
        print("Passageway", i, info, flush=True)
    if (scene / "metric_rectification.json").exists():
        manifest["plane_rectification_file"] = "metric_rectification.json"
    (scene / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.skip_lab_timeline:
        from concurrent.futures import ThreadPoolExecutor

        def download_one(i):
            item = timeline(root / f"cam{i}", BASE + f"4p-c{i}.avi", duration=45)
            item["camera"] = f"cam{i}"
            print("Laboratory timeline", i, flush=True)
            return item

        with ThreadPoolExecutor(max_workers=4) as executor:
            details = list(executor.map(download_one, range(4)))
        (root / "timeline_manifest.json").write_text(
            json.dumps({"source": PAGE, "cameras": details}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
