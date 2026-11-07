#!/usr/bin/env python3
"""Prepare real CITIC capture1; fetch only its ZIP members using HTTP ranges and CRCs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.geometry import save_json

REPO = "https://github.com/GTEC-UDC/citic-mtmc-dataset"
RECORD = "https://zenodo.org/api/records/17511785"
CAMERAS = {"HallWide": "entrance_hall", "HallSeg": "entrance_hall",
           "PF": "passageway_pf", "PC": "passageway_pc", "Showroom": "showroom"}


def fetch(url, target, byte_range=None):
    command = ["curl", "-L", "--fail", "--silent", "--show-error", "--retry", "3",
               "--connect-timeout", "30", "--max-time", "900"]
    if byte_range:
        command += ["--range", byte_range]
    temporary = target.with_suffix(target.suffix + ".part")
    subprocess.run(command + ["-o", str(temporary), url], check=True)
    temporary.replace(target)


def zip_index(tail, total_size):
    """Read a standard ZIP central directory without downloading unrelated captures."""
    end = tail.rfind(b"PK\x05\x06")
    if end < 0:
        raise ValueError("Missing ZIP end record")
    _, disk, central_disk, _, count, size, offset, _ = struct.unpack_from("<4s4H2LH", tail, end)
    if disk or central_disk or offset < total_size - len(tail):
        raise ValueError("Unsupported ZIP index layout")
    central = tail[offset - (total_size - len(tail)):][:size]
    # ZipFile adjusts header offsets when the body is absent; retain true offsets separately.
    result = []
    cursor = 0
    for _ in range(count):
        h = struct.unpack_from("<4s6H3L5H2L", central, cursor)
        if h[0] != b"PK\x01\x02":
            raise ValueError("Invalid ZIP directory")
        n, extra, comment = h[10:13]
        name = central[cursor + 46:cursor + 46 + n].decode("utf-8")
        result.append(dict(name=name, method=h[4], crc=h[7], compressed=h[8],
                           size=h[9], offset=h[-1], name_length=n))
        cursor += 46 + n + extra + comment
    return result


def extract_member(entry, url, root):
    target = root / "capture1" / Path(entry["name"]).name
    if target.exists() and target.stat().st_size == entry["size"]:
        if zlib.crc32(target.read_bytes()) == entry["crc"]:
            return target
    cache = root / "ranges"
    cache.mkdir(exist_ok=True)
    header_file = cache / (target.stem + ".header")
    start = entry["offset"]
    prefix = root / "capture_archive.zip"
    if prefix.exists() and prefix.stat().st_size >= start + 30:
        with prefix.open("rb") as stream:
            stream.seek(start)
            header = stream.read(30)
    else:
        fetch(url, header_file, f"{start}-{start + 29}")
        header = header_file.read_bytes()
    if len(header) != 30 or header[:4] != b"PK\x03\x04":
        raise ValueError("Server did not return the requested ZIP header range")
    n, extra = struct.unpack_from("<HH", header, 26)
    begin = start + 30 + n + extra
    end = begin + entry["compressed"]
    if prefix.exists() and prefix.stat().st_size >= end:
        with prefix.open("rb") as stream:
            stream.seek(begin)
            compressed = stream.read(entry["compressed"])
    else:
        payload = cache / (target.stem + ".deflate")
        if not payload.exists() or payload.stat().st_size != entry["compressed"]:
            fetch(url, payload, f"{begin}-{end - 1}")
        compressed = payload.read_bytes()
    if len(compressed) != entry["compressed"]:
        raise ValueError("HTTP range size mismatch")
    data = zlib.decompress(compressed, -15) if entry["method"] == 8 else compressed
    if len(data) != entry["size"] or zlib.crc32(data) != entry["crc"]:
        raise ValueError(f"ZIP CRC mismatch: {target.name}")
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix(".part")
    temporary.write_bytes(data)
    temporary.replace(target)
    print(f"Verified capture1/{target.name}: {len(data):,} bytes", flush=True)
    return target


def prepare(root):
    metadata = root / "metadata"
    if not metadata.exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(metadata)], check=True)
    record_file = root / "zenodo_record.json"
    if not record_file.exists():
        fetch(RECORD, record_file)
    record = json.loads(record_file.read_text())
    archive = next(f for f in record["files"] if f["key"].endswith(".zip"))
    url, size = archive["links"]["self"], archive["size"]
    tail_file = root / "zip_index.bin"
    if not tail_file.exists():
        fetch(url, tail_file, f"{size - 65536}-{size - 1}")
    index = zip_index(tail_file.read_bytes(), size)
    checked_metadata = []
    for entry in index:
        relative = Path(*Path(entry["name"]).parts[1:])
        if (relative.as_posix() == "capture1_gt.mot.txt" or
                (relative.parent.as_posix() == "map_config" and relative.suffix == ".json")):
            path = metadata / relative
            if path.stat().st_size != entry["size"] or zlib.crc32(path.read_bytes()) != entry["crc"]:
                raise ValueError(f"Git metadata differs from archived video release: {relative}")
            checked_metadata.append(str(relative))
    entries = [e for e in index
               if "/capture1/" in e["name"] and e["name"].endswith(".mp4")]
    if {Path(e["name"]).stem for e in entries} != set(CAMERAS):
        raise ValueError("Archive must contain exactly the five capture1 cameras")
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda e: extract_member(e, url, root), entries))
    cameras = []
    for name, room in CAMERAS.items():
        path = root / "capture1" / (name + ".mp4")
        cap = cv2.VideoCapture(str(path))
        info = dict(id=name, room_id=room, video=str(path.relative_to(root)),
                    fps=cap.get(cv2.CAP_PROP_FPS), frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        ok, _ = cap.read()
        cap.release()
        if not ok or info["fps"] != 25 or info["frames"] < 4200:
            raise ValueError(f"Unexpected capture profile: {info}")
        cameras.append(info)
    commit = subprocess.check_output(["git", "-C", str(metadata), "rev-parse", "HEAD"], text=True).strip()
    manifest = dict(dataset="CITIC-MTMC", sequence="capture1", source=REPO,
                    license="CC BY-NC-SA 4.0", metadata_commit=commit,
                    metadata_matches_archive=checked_metadata,
                    synchronized=True, coordinate_units="meters", cameras=cameras,
                    metadata_dir="metadata", ground_truth="metadata/capture1_gt.mot.txt",
                    source_frame_convention="video zero-based; MOT ground truth one-based",
                    archive=dict(url=url, size=size, checksum=archive["checksum"],
                                 verification="individual ZIP member CRC32 and uncompressed size"))
    save_json(root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/citic_mtmc"))
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    prepare(args.data_dir)


if __name__ == "__main__":
    main()
