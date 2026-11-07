#!/usr/bin/env python3
"""Obtain real Passageway frames 250..349 for closer pedestrian verification."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import cv2
from download_sample_data import fetch


def main():
    root = Path("data/passageway")
    manifest = json.loads((root / "manifest.json").read_text())

    def one(camera):
        folder = root / camera["id"]
        frames = folder / "verification_frames"
        frames.mkdir(exist_ok=True)
        video = folder / "verification.avi"
        if not video.exists():
            with tempfile.TemporaryDirectory(prefix="passageway-verification-") as temp:
                prefix = Path(temp) / "prefix.avi"
                fetch(camera["source_url"], prefix, 8 * 1024**2)
                cap = cv2.VideoCapture(str(prefix))
                writer = cv2.VideoWriter(
                    str(video),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    camera["fps"],
                    (camera["width"], camera["height"]),
                )
                try:
                    for index in range(350):
                        ok, frame = cap.read()
                        if not ok:
                            raise RuntimeError(
                                f'Prefix too short for {camera["id"]}, frame {index}'
                            )
                        if index >= 250:
                            if not cv2.imwrite(str(frames / f"{index-250:06d}.jpg"), frame):
                                raise RuntimeError("Cannot write frame")
                            writer.write(frame)
                finally:
                    cap.release()
                    writer.release()
        camera.update(
            verification_video=str(video.relative_to(root)),
            verification_frames_dir=str(frames.relative_to(root)),
            verification_start_frame=250,
        )
        print("Passageway later clip", camera["id"], flush=True)
        return camera

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest["cameras"] = list(pool.map(one, manifest["cameras"]))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
