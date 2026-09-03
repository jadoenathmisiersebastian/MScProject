from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUIRED_TOP_LEVEL_KEYS = {"trial_id", "scene_id", "frame_id", "image_path", "object", "candidate_grasp", "outcome"}
REQUIRED_OBJECT_KEYS = {"class_name", "instance_id", "bbox_xyxy", "image_center"}
REQUIRED_CANDIDATE_KEYS = {"candidate_id", "grasp_type", "wrist_roll_degrees", "hand_aperture", "approach_direction_camera"}
REQUIRED_OUTCOME_KEYS = {"success", "success_score"}


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_number_list(value: Any, length: int, path: str, errors: list[str]) -> None:
    if not isinstance(value, list) or len(value) != length:
        errors.append(f"{path} must be a list of length {length}")
        return
    for index, item in enumerate(value):
        if not _is_number(item):
            errors.append(f"{path}[{index}] must be numeric")


def validate_grasp_trial(record: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []

    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(record))
    if missing:
        errors.append(f"Missing top-level keys: {missing}")

    obj = record.get("object", {})
    if not isinstance(obj, dict):
        errors.append("object must be a dictionary")
        obj = {}
    else:
        missing_object = sorted(REQUIRED_OBJECT_KEYS - set(obj))
        if missing_object:
            errors.append(f"Missing object keys: {missing_object}")

    candidate = record.get("candidate_grasp", {})
    if not isinstance(candidate, dict):
        errors.append("candidate_grasp must be a dictionary")
        candidate = {}
    else:
        missing_candidate = sorted(REQUIRED_CANDIDATE_KEYS - set(candidate))
        if missing_candidate:
            errors.append(f"Missing candidate_grasp keys: {missing_candidate}")

    outcome = record.get("outcome", {})
    if not isinstance(outcome, dict):
        errors.append("outcome must be a dictionary")
        outcome = {}
    else:
        missing_outcome = sorted(REQUIRED_OUTCOME_KEYS - set(outcome))
        if missing_outcome:
            errors.append(f"Missing outcome keys: {missing_outcome}")

    if "bbox_xyxy" in obj:
        _check_number_list(obj["bbox_xyxy"], 4, "object.bbox_xyxy", errors)
    if "image_center" in obj:
        _check_number_list(obj["image_center"], 2, "object.image_center", errors)

    object_pose = obj.get("object_pose_camera")
    if object_pose is not None:
        if not isinstance(object_pose, dict):
            errors.append("object.object_pose_camera must be a dictionary")
        else:
            if "position" in object_pose:
                _check_number_list(object_pose["position"], 3, "object.object_pose_camera.position", errors)
            if "rotation_quat" in object_pose:
                _check_number_list(object_pose["rotation_quat"], 4, "object.object_pose_camera.rotation_quat", errors)

    if "dimensions_m" in obj:
        _check_number_list(obj["dimensions_m"], 3, "object.dimensions_m", errors)

    if "grasp_type" in candidate and not isinstance(candidate["grasp_type"], str):
        errors.append("candidate_grasp.grasp_type must be a string")
    if "wrist_roll_degrees" in candidate and not _is_number(candidate["wrist_roll_degrees"]):
        errors.append("candidate_grasp.wrist_roll_degrees must be numeric")
    if "hand_aperture" in candidate:
        aperture = candidate["hand_aperture"]
        if not _is_number(aperture) or not 0.0 <= float(aperture) <= 1.0:
            errors.append("candidate_grasp.hand_aperture must be numeric in [0, 1]")
    if "approach_direction_camera" in candidate:
        _check_number_list(candidate["approach_direction_camera"], 3, "candidate_grasp.approach_direction_camera", errors)

    if "success" in outcome and not isinstance(outcome["success"], bool):
        errors.append("outcome.success must be boolean")
    if "success_score" in outcome:
        score = outcome["success_score"]
        if not _is_number(score) or not 0.0 <= float(score) <= 1.0:
            errors.append("outcome.success_score must be numeric in [0, 1]")

    optional_non_negative = ["lift_height_m", "hold_duration_s", "slip_distance_m"]
    for key in optional_non_negative:
        if key in outcome and (not _is_number(outcome[key]) or float(outcome[key]) < 0.0):
            errors.append(f"outcome.{key} must be a non-negative number")

    if "pregrasp_collision" in outcome and not isinstance(outcome["pregrasp_collision"], bool):
        errors.append("outcome.pregrasp_collision must be boolean")

    return ValidationResult(valid=not errors, errors=errors)
