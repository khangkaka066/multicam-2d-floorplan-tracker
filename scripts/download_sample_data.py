#!/usr/bin/env python3
"""Download bounded EPFL Laboratory video prefixes and extract synchronized frames."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import cv2

PAGE = "https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/"
BASE = "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/"
CAL = "https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/calibration-{}.txt"


def fetch(url, target, byte_limit=None):
    command = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "2",
        "--connect-timeout",
        "20",
        "--max-time",
        "180",
    ]
    if byte_limit:
        command += ["--range", f"0-{byte_limit - 1}", "--max-filesize", str(byte_limit)]
    subprocess.run(command + ["--output", str(target), url], check=True)


def extract(prefix, video, frames_dir, count):
    capture = cv2.VideoCapture(str(prefix))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or fps <= 0:
        raise RuntimeError(f"Cannot decode {prefix}")
    writer = None
    extracted = 0
    try:
        for index in range(count):
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(video), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError("MJPG video writer unavailable")
            if not cv2.imwrite(str(frames_dir / f"{index:06d}.jpg"), frame):
                raise RuntimeError("Cannot write frame")
            writer.write(frame)
            extracted += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if extracted != count:
        raise RuntimeError(f"Prefix decoded {extracted}/{count} frames; increase --prefix-mb")
    return {"width": width, "height": height, "fps": fps, "frames": extracted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data"
    )
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--sequence", choices=["4p", "6p"], default="4p")
    parser.add_argument("--prefix-mb", type=int, default=8)
    parser.add_argument(
        "--verification-start",
        type=int,
        default=500,
        help="Additional synchronized source frame interval; -1 disables",
    )
    args = parser.parse_args()
    if args.frames < 1 or not 1 <= args.prefix_mb <= 64:
        parser.error("frames must be positive; prefix-mb must be 1..64")
    root = args.data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("sequence") != args.sequence or any(
            c["frames"] != args.frames for c in previous["cameras"]
        ):
            raise RuntimeError("Use a separate --data-dir for another sequence or frame count")
    calibration = root / f"calibration-{args.sequence}.txt"
    fetch(CAL.format(args.sequence), calibration)
    manifest = {
        "dataset": "EPFL Laboratory",
        "sequence": args.sequence,
        "source": PAGE,
        "license": "CVLab EPFL copyright; research use only; cite Fleuret et al. TPAMI 2008",
        "calibration": calibration.name,
        "calibration_image_size": [360, 288],
        "coordinate_units": "EPFL top-view units (not validated meters)",
        "room_id": "laboratory",
        "synchronized": True,
        "first_source_frame": 0,
        "cameras": [],
    }
    for camera in range(4):
        folder = root / f"cam{camera}"
        frames = folder / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        video = folder / "sample.avi"
        source_url = BASE + f"{args.sequence}-c{camera}.avi"
        reusable = None
        if video.exists():
            cap = cv2.VideoCapture(str(video))
            if (
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == args.frames
                and len(list(frames.glob("*.jpg"))) == args.frames
            ):
                reusable = {
                    "width": int(cap.get(3)),
                    "height": int(cap.get(4)),
                    "fps": cap.get(5),
                    "frames": args.frames,
                }
            cap.release()
        if reusable:
            info = reusable
        else:
            with tempfile.TemporaryDirectory(prefix="epfl-") as temporary:
                prefix = Path(temporary) / "prefix.avi"
                fetch(source_url, prefix, args.prefix_mb * 1024**2)
                info = extract(prefix, video, frames, args.frames)
        info.update(
            {
                "id": f"cam{camera}",
                "room_id": "laboratory",
                "source_url": source_url,
                "video": str(video.relative_to(root)),
                "frames_dir": str(frames.relative_to(root)),
                "sample_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            }
        )
        if args.verification_start >= 0:
            verification = folder / "verification.avi"
            verification_frames = folder / "verification_frames"
            verification_frames.mkdir(exist_ok=True)
            old_camera = (
                next((c for c in previous["cameras"] if c["id"] == info["id"]), {})
                if manifest_path.exists()
                else {}
            )
            if (
                not verification.exists()
                or old_camera.get("verification_start_frame") != args.verification_start
            ):
                partial = folder / "verification.part.avi"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-rw_timeout",
                        "30000000",
                        "-ss",
                        str(args.verification_start / info["fps"]),
                        "-i",
                        source_url,
                        "-an",
                        "-frames:v",
                        str(args.frames),
                        "-c:v",
                        "mjpeg",
                        str(partial),
                    ],
                    check=True,
                    timeout=240,
                )
                partial.replace(verification)
            with tempfile.TemporaryDirectory(prefix="epfl-verification-") as temporary:
                copied = Path(temporary) / "copy.avi"
                extract(verification, copied, verification_frames, args.frames)
            info.update(
                {
                    "verification_video": str(verification.relative_to(root)),
                    "verification_frames_dir": str(verification_frames.relative_to(root)),
                    "verification_start_frame": args.verification_start,
                }
            )
        manifest["cameras"].append(info)
        print(f"cam{camera}: {info}", flush=True)
    temporary_manifest = root / "manifest.json.part"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(manifest_path)
    print(f"Saved {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
