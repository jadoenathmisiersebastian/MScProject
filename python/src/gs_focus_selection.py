from __future__ import annotations

from math import sqrt


def focus_distance(frame: dict, detection: dict) -> float:
    image_width = frame["image_width"]
    image_height = frame["image_height"]

    image_center_x = image_width / 2
    image_center_y = image_height / 2

    object_center_x, object_center_y = detection["image_center"]

    dx = (object_center_x - image_center_x) / image_width
    dy = (object_center_y - image_center_y) / image_height

    return sqrt(dx ** 2 + dy ** 2)


def select_focused_detection(
    frame: dict,
    min_confidence: float = 0.25,
    max_focus_distance: float = 0.25,
) -> dict | None:
    detections = [
        det for det in frame["detections"]
        if det["confidence"] >= min_confidence
    ]

    if not detections:
        return None

    selected = min(
        detections,
        key=lambda det: focus_distance(frame, det),
    )

    distance = focus_distance(frame, selected)

    if distance > max_focus_distance:
        return None

    selected = dict(selected)
    selected["focus_distance"] = distance

    return selected