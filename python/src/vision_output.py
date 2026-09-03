from __future__ import annotations

from pathlib import Path
import json
from math import sqrt


def focus_distance(frame: dict, detection: dict) -> float:
    image_width = frame["image_width"]
    image_height = frame["image_height"]

    image_center_x = image_width / 2
    image_center_y = image_height / 2

    object_center_x, object_center_y = detection["image_center"]

    dx = (object_center_x - image_center_x) / image_width
    dy = (object_center_y - image_center_y) / image_height

    return sqrt(dx**2 + dy**2)


def bbox_area_normalized(detection: dict, image_width: int, image_height: int) -> float:
    x1, y1, x2, y2 = detection["bbox_xyxy"]

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return (width * height) / float(image_width * image_height)


def build_vision_frame(
    frame: dict,
    min_confidence: float = 0.25,
    max_focus_distance: float = 0.25,
) -> dict:
    image_width = frame["image_width"]
    image_height = frame["image_height"]

    valid_detections = [
        det for det in frame["detections"]
        if det["confidence"] >= min_confidence
    ]

    focused_detection_id = None

    if valid_detections:
        selected = min(
            valid_detections,
            key=lambda det: focus_distance(frame, det),
        )

        selected_focus_distance = focus_distance(frame, selected)

        if selected_focus_distance <= max_focus_distance:
            focused_detection_id = selected["detection_id"]

    objects = []

    for detection in frame["detections"]:
        distance = focus_distance(frame, detection)

        objects.append({
            "detection_id": detection["detection_id"],
            "class_id": detection["class_id"],
            "class_name": detection["class_name"],
            "confidence": detection["confidence"],

            "bbox_xyxy": detection["bbox_xyxy"],
            "bbox_xywh": detection["bbox_xywh"],
            "image_center": detection["image_center"],
            "normalized_center": detection["normalized_center"],

            "bbox_area_pixels": detection["area_pixels"],
            "bbox_area_normalized": bbox_area_normalized(
                detection,
                image_width,
                image_height,
            ),

            "focus_distance": distance,
            "is_focused_object": detection["detection_id"] == focused_detection_id,
        })

    focused_object = None

    for obj in objects:
        if obj["is_focused_object"]:
            focused_object = obj
            break

    return {
        "frame_id": frame["frame_id"],
        "image_path": frame["image_path"],
        "image_width": image_width,
        "image_height": image_height,
        "focused_object": focused_object,
        "objects": objects,
    }


def build_vision_output(
    detections_payload: dict,
    min_confidence: float = 0.25,
    max_focus_distance: float = 0.25,
) -> dict:
    return {
        "source_detections": detections_payload.get("input_dir"),
        "selection_method": "closest_bbox_center_to_image_center",
        "min_confidence": min_confidence,
        "max_focus_distance": max_focus_distance,
        "frames": [
            build_vision_frame(
                frame,
                min_confidence=min_confidence,
                max_focus_distance=max_focus_distance,
            )
            for frame in detections_payload["frames"]
        ],
    }


def run_vision_output_export(
    detections_path: str | Path,
    output_path: str | Path | None = None,
    min_confidence: float = 0.25,
    max_focus_distance: float = 0.25,
) -> Path:
    detections_path = Path(detections_path).expanduser().resolve()

    if output_path is None:
        output_path = detections_path.with_name("vision_output.json")
    else:
        output_path = Path(output_path).expanduser().resolve()

    with detections_path.open("r") as f:
        detections_payload = json.load(f)

    vision_output = build_vision_output(
        detections_payload=detections_payload,
        min_confidence=min_confidence,
        max_focus_distance=max_focus_distance,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(vision_output, f, indent=2)

    return output_path