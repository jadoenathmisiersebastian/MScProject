from __future__ import annotations

from pathlib import Path
import json

from .gs_focus_selection import select_focused_detection


GRASP_RULES = {
    "bottle": {
        "grasp_type": "cylindrical",
        "approach_direction": "side",
        "priority": 1,
    },
    "drink_carton": {
        "grasp_type": "power",
        "approach_direction": "side",
        "priority": 2,
    },
    "food_box": {
        "grasp_type": "power",
        "approach_direction": "front_or_top",
        "priority": 2,
    },
    "glass": {
        "grasp_type": "cylindrical",
        "approach_direction": "side_or_top",
        "priority": 1,
    },
}


def _recommend_pose(detection: dict) -> dict:
    class_name = detection["class_name"]

    rule = GRASP_RULES.get(
        class_name,
        {
            "grasp_type": "unknown",
            "approach_direction": "unknown",
            "priority": 99,
        },
    )

    return {
        "grasp_type": rule["grasp_type"],
        "approach_direction": rule["approach_direction"],
        "priority": rule["priority"],
        "reason": (
            f"{class_name} is the focused object "
            f"with confidence {detection['confidence']:.2f} "
            f"and focus distance {detection['focus_distance']:.3f}"
        ),
    }


def run_pose_selection(
    detections_path: str | Path,
    output_path: str | Path | None = None,
    min_confidence: float = 0.25,
    max_focus_distance: float = 0.25,
) -> Path:
    detections_path = Path(detections_path).expanduser().resolve()

    if output_path is None:
        output_path = detections_path.with_name("pose_selection.json")
    else:
        output_path = Path(output_path).expanduser().resolve()

    with open(detections_path, "r") as f:
        detection_payload = json.load(f)

    frame_outputs = []

    for frame in detection_payload["frames"]:
        selected = select_focused_detection(
            frame=frame,
            min_confidence=min_confidence,
            max_focus_distance=max_focus_distance,
        )

        if selected is None:
            frame_outputs.append(
                {
                    "frame_id": frame["frame_id"],
                    "image_path": frame["image_path"],
                    "selected_object": None,
                    "pose_selection": None,
                    "status": "no_focused_object",
                }
            )
            continue

        frame_outputs.append(
            {
                "frame_id": frame["frame_id"],
                "image_path": frame["image_path"],
                "selected_object": selected,
                "pose_selection": _recommend_pose(selected),
                "status": "focused_object_selected",
            }
        )

    payload = {
        "source_detections": str(detections_path),
        "selection_method": "closest_bbox_center_to_image_center",
        "min_confidence": min_confidence,
        "max_focus_distance": max_focus_distance,
        "frames": frame_outputs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    return output_path
