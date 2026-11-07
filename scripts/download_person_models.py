#!/usr/bin/env python3
"""Fetch official model files required by the person-projection stage."""
from pathlib import Path
import hashlib
import json
import subprocess


def main():
    root = Path(__file__).resolve().parents[1] / "models"
    root.mkdir(exist_ok=True)
    sources = []
    for filename, url, prefix in [
        (
            "yolo11n-pose.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt",
            None,
        ),
        (
            "resnet18-f37072fd.pth",
            "https://download.pytorch.org/models/resnet18-f37072fd.pth",
            "f37072fd",
        ),
    ]:
        path = root / filename
        if not path.exists():
            part = path.with_suffix(path.suffix + ".part")
            subprocess.run(
                ["curl", "-fL", "--retry", "2", "--max-time", "180", "-o", str(part), url],
                check=True,
            )
            part.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if prefix and not digest.startswith(prefix):
            raise RuntimeError(f"Unexpected hash for {filename}")
        sources.append({"file": filename, "url": url, "sha256": digest})
    (root / "person_model_sources.json").write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
