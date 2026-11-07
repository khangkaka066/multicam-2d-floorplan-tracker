#!/usr/bin/env python3
"""Download pinned official inference code and model weights into models/."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VENDOR_COMMIT = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
SEG_REVISION = "de01bae28967510f9ddd496c60a969357195400c"
DEPTH_HASH = "b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545"


def download(url, target, expected=None):
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".part")
        subprocess.run(
            ["curl", "-fL", "--retry", "2", "--max-time", "240", "-o", str(partial), url],
            check=True,
        )
        partial.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if expected and digest != expected:
        raise ValueError(f"Unexpected checkpoint hash: {target}")
    return {"url": url, "sha256": digest, "file": str(target.relative_to(ROOT))}


def main():
    folder = ROOT / "models"
    folder.mkdir(exist_ok=True)
    vendor = folder / "Depth-Anything-V2"
    if not vendor.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/DepthAnything/Depth-Anything-V2.git", str(vendor)],
            check=True,
        )
    current = subprocess.check_output(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
    ).strip()
    if current != VENDOR_COMMIT:
        if subprocess.check_output(
            ["git", "-C", str(vendor), "status", "--porcelain"], text=True
        ).strip():
            raise RuntimeError("Vendor checkout has local changes; cannot switch revisions")
        subprocess.run(["git", "-C", str(vendor), "checkout", VENDOR_COMMIT], check=True)
    records = [
        download(
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth",
            folder / "depth_anything_v2_metric_hypersim_vits.pth",
            DEPTH_HASH,
        )
    ]
    segmenter = folder / "segformer-b2"
    segmenter.mkdir(exist_ok=True)
    for filename in ["config.json", "preprocessor_config.json", "pytorch_model.bin"]:
        records.append(
            download(
                f"https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512/resolve/{SEG_REVISION}/{filename}",
                segmenter / filename,
            )
        )
    (folder / "structural_sources.json").write_text(
        json.dumps({"vendor_commit": VENDOR_COMMIT, "models": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
