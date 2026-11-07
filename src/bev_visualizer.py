from typing import Protocol
import cv2
import numpy as np
from .geometry import project
from .map_builder import COLORS


class BEVVisualizer(Protocol):
    def render(self, dataset, record, map_image, config): ...


class PanelVisualizer:
    def render(self, dataset, record, map_image, config):
        bev = map_image.copy()
        panels = []
        matrix = config["T_map_to_canvas"]
        skipped = 0
        for index, camera in enumerate(dataset.cameras):
            frame = dataset.frame(camera, record["index"], verification=True)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 24), (240, 240, 240), -1)
            source_frame = (
                camera["verification_start_frame"] + record["index"]
                if "verification_start_frame" in camera
                else record["source_frame"]
            )
            cv2.putText(
                frame, f"{camera['id']} | frame {source_frame}", (5, 17), 0, 0.43, (30, 30, 30), 1
            )
            for detection in record["detections"]:
                if detection["camera"] != camera["id"]:
                    continue
                color = COLORS[index % 4]
                x1, y1, x2, y2 = np.rint(detection["bbox"]).astype(int)
                uv = tuple(np.rint(detection["uv"]).astype(int))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.circle(frame, uv, 4, color, -1)
                label = (
                    f"ID {detection['associated_id']} "
                    if detection.get("associated_id") is not None
                    else ""
                ) + detection["method"]
                cv2.putText(frame, label, (x1, max(35, y1 - 5)), 0, 0.32, color, 1)
                room = next(r for r in config["rooms"] if r["id"] == detection["room_id"])
                clearance = cv2.pointPolygonTest(
                    np.asarray(room["polygon"], np.float32), tuple(detection["xy"]), True
                )
                # Reject invalid projections from the architectural overlay. Never move them to a wall.
                if clearance < config["minimum_display_clearance_m"]:
                    skipped += 1
                    continue
                point = tuple(np.rint(project(matrix, [detection["xy"]])[0]).astype(int))
                if not config.get("render_fused_people"):
                    cv2.circle(bev, point, 7, (249, 248, 244), -1, cv2.LINE_AA)
                    cv2.circle(bev, point, 5, color, -1, cv2.LINE_AA)
                elif config.get("show_raw_projections"):
                    overlay = bev.copy()
                    cv2.circle(overlay, point, 2, color, -1, cv2.LINE_AA)
                    person = next(
                        (
                            p
                            for p in record.get("fused_people", [])
                            if p["id"] == detection.get("associated_id")
                        ),
                        None,
                    )
                    if person:
                        end = tuple(np.int32(project(matrix, [person["xy"]])[0]))
                        cv2.line(overlay, point, end, color, 1, cv2.LINE_AA)
                    bev = cv2.addWeighted(bev, 0.75, overlay, 0.25, 0)
            panels.append(frame)
        if config.get("render_fused_people"):
            for person in record.get("fused_people", []):
                point = tuple(np.rint(project(matrix, [person["xy"]])[0]).astype(int))
                color = [
                    (160, 70, 20),
                    (40, 130, 65),
                    (175, 90, 140),
                    (40, 120, 190),
                    (150, 100, 40),
                    (110, 65, 160),
                ][(person["id"] - 1) % 6]
                cv2.circle(bev, point, 10, (250, 250, 250), -1, cv2.LINE_AA)
                cv2.circle(bev, point, 8, color, -1, cv2.LINE_AA)
                cv2.circle(bev, point, 8, (45, 45, 35), 1, cv2.LINE_AA)
                text = f"ID {person['id']}" + (
                    "*" if person.get("state") == "predicted_short_occlusion" else ""
                )
                size = cv2.getTextSize(text, 0, 0.48, 1)[0]
                start = (point[0] + 12, point[1] - 9)
                cv2.rectangle(
                    bev,
                    (start[0] - 3, start[1] - size[1] - 3),
                    (start[0] + size[0] + 3, start[1] + 4),
                    (250, 250, 250),
                    -1,
                )
                cv2.putText(bev, text, start, 0, 0.48, color, 1, cv2.LINE_AA)
            for room in config["rooms"]:
                p = np.array(room["polygon"])
                center = np.array([p[:, 0].mean(), p[:, 1].max() - 0.55])
                pixel = np.int32(project(matrix, [center])[0])
                count = record.get("headcounts", {}).get(room["id"], 0)
                text = f"{room['id'].upper()}: {count} people"
                size = cv2.getTextSize(text, 0, 0.48, 1)[0]
                start = (int(pixel[0] - size[0] / 2), int(pixel[1]))
                cv2.rectangle(
                    bev,
                    (start[0] - 8, start[1] - size[1] - 8),
                    (start[0] + size[0] + 8, start[1] + 7),
                    (252, 252, 249),
                    -1,
                )
                cv2.putText(bev, text, start, 0, 0.48, (40, 45, 35), 1, cv2.LINE_AA)
        cv2.putText(
            bev,
            record.get(
                "frame_note",
                f"Source frame {record['source_frame']} | rejected near/outside wall: {skipped}",
            ),
            (35, 88),
            0,
            0.40,
            (100, 90, 75),
            1,
            cv2.LINE_AA,
        )
        rows = [
            np.hstack(
                panels[i : i + 2]
                if len(panels[i : i + 2]) == 2
                else [panels[i], np.full_like(panels[i], 247)]
            )
            for i in range(0, len(panels), 2)
        ]
        mosaic = np.vstack(rows)
        output = np.full(
            (max(mosaic.shape[0], bev.shape[0]), mosaic.shape[1] + bev.shape[1], 3), 247, np.uint8
        )
        output[: mosaic.shape[0], : mosaic.shape[1]] = mosaic
        output[: bev.shape[0], mosaic.shape[1] :] = bev
        return output


def export_verification(dataset, records, map_image, config, output):
    from pathlib import Path

    output = Path(output)
    visualizer = PanelVisualizer()
    first = visualizer.render(dataset, records[0], map_image, config)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output / "verification_projection.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        dataset.cameras[0]["fps"],
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Cannot create verification video")
    best = max(
        records,
        key=lambda r: (
            sum(
                d.get("display_accepted", False) and d["method"] == "ankles"
                for d in r["detections"]
            ),
            sum(d["confidence"] for d in r["detections"] if d.get("display_accepted", False)),
        ),
    )
    if config.get("verification_record_index") is not None:
        best = next((r for r in records if r["index"] == config["verification_record_index"]), best)
    try:
        for record in records:
            image = visualizer.render(dataset, record, map_image, config)
            writer.write(image)
            if record is best:
                mosaic_width = image.shape[1] - map_image.shape[1]
                if config.get("render_fused_people"):
                    cv2.imwrite(
                        str(output / "fused_bev.png"), image[: map_image.shape[0], mosaic_width:]
                    )
                for name in ["verification_bev.png", "verification_projection.png"]:
                    if not cv2.imwrite(str(output / name), image):
                        raise RuntimeError("Cannot save verification image")
    finally:
        writer.release()
