#!/usr/bin/env python3
"""Render an explicitly synthetic three-room topology showcase; uses no EPFL imagery."""
from pathlib import Path
import sys
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.architecture import room_from_polygon, layout_rooms
from src.topology import declared_doors, load_topology
from src.map_builder import configure_canvas, draw_floorplan


def main():
    root = Path(__file__).resolve().parents[1]
    rooms = [
        room_from_polygon(name, [[0, 0], [4, 0], [4, 4], [0, 4]])
        for name in ["room1", "room2", "room3"]
    ]
    rooms[2] = room_from_polygon("room3", [[0, 0], [4, 0], [4, 4], [2, 4], [2, 2], [0, 2]])
    for room, walls in zip(rooms, [[1], [3, 1], [0]]):
        declared_doors(
            room,
            [
                {"id": f"D{i+1:02d}", "wall_index": wall, "offset_m": 1.5, "width_m": 1.0}
                for i, wall in enumerate(walls)
            ],
        )
    rooms, alignment = layout_rooms(rooms, load_topology(root / "configs/topology.example.json"))
    config = {
        "rooms": rooms,
        "alignment": alignment,
        "source": "SYNTHETIC TOPOLOGY FIXTURE",
        "subtitle": "SYNTHETIC TEST | 3 rooms | 90-degree rotation | 0.2 m doorway connector",
    }
    config.update(configure_canvas(rooms))
    output = root / "assets/topology_stitching_demo.png"
    if not cv2.imwrite(str(output), draw_floorplan(config)):
        raise RuntimeError("Cannot write topology demo")
    print(output)


if __name__ == "__main__":
    main()
