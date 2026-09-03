from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
from typing import Any, Iterable


SUPPORTED_EXTENSIONS = {".jsonl", ".ndjson", ".json"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_number_list(value: Any, length: int, path: str, errors: list[str]) -> None:
    if not isinstance(value, list) or len(value) != length:
        errors.append(f"{path} must be a list of length {length}")
        return

    for index, item in enumerate(value):
        if not _is_number(item):
            errors.append(f"{path}[{index}] must be numeric")


def validate_vision_label_record(
    record: dict[str, Any],
    require_semantic_class: bool = False,
) -> list[str]:
    errors: list[str] = []

    required_top_level = {"frame_id", "scene_id", "image_path", "camera", "objects"}
    missing_top_level = sorted(required_top_level - set(record))

    if missing_top_level:
        errors.append(f"Missing top-level keys: {missing_top_level}")

    camera = record.get("camera", {})

    if not isinstance(camera, dict):
        errors.append("camera must be a dictionary")
        camera = {}

    required_camera = {
        "position_world",
        "rotation_world_quat",
        "image_width",
        "image_height",
        "field_of_view_degrees",
    }

    missing_camera = sorted(required_camera - set(camera))

    if missing_camera:
        errors.append(f"Missing camera keys: {missing_camera}")

    if "position_world" in camera:
        _check_number_list(camera["position_world"], 3, "camera.position_world", errors)

    if "rotation_world_quat" in camera:
        _check_number_list(camera["rotation_world_quat"], 4, "camera.rotation_world_quat", errors)

    for key in ["image_width", "image_height"]:
        if key in camera:
            value = camera[key]
            if not isinstance(value, int) or value <= 0:
                errors.append(f"camera.{key} must be a positive integer")

    if "field_of_view_degrees" in camera:
        value = camera["field_of_view_degrees"]
        if not _is_number(value) or value <= 0:
            errors.append("camera.field_of_view_degrees must be a positive number")

    objects = record.get("objects", [])
    is_null_sample = record.get("is_null_sample", False)

    if not isinstance(is_null_sample, bool):
        errors.append("is_null_sample must be boolean when present")
        is_null_sample = False

    if not isinstance(objects, list):
        errors.append("objects must be a list")
        objects = []

    required_object = {
        "object_id",
        "object_name",
        "class_name",
        "bbox_xyxy",
        "image_center",
        "normalized_center",
        "bbox_area_pixels",
        "bbox_area_normalized",
        "position_camera",
        "distance_camera_m",
        "dimensions_m",
        "is_in_front_of_camera",
        "focus_distance",
        "is_focused_object",
    }

    focused_count = 0

    for object_index, obj in enumerate(objects):
        path = f"objects[{object_index}]"

        if not isinstance(obj, dict):
            errors.append(f"{path} must be a dictionary")
            continue

        missing_object = sorted(required_object - set(obj))

        if missing_object:
            errors.append(f"{path} missing keys: {missing_object}")

        semantic_class = obj.get("semantic_class")
        if require_semantic_class and not isinstance(semantic_class, str):
            errors.append(f"{path}.semantic_class is required and must be a string")
        elif semantic_class is not None and (not isinstance(semantic_class, str) or not semantic_class.strip()):
            errors.append(f"{path}.semantic_class must be a non-empty string when present")

        if "bbox_xyxy" in obj:
            _check_number_list(obj["bbox_xyxy"], 4, f"{path}.bbox_xyxy", errors)

        if "image_center" in obj:
            _check_number_list(obj["image_center"], 2, f"{path}.image_center", errors)

        if "normalized_center" in obj:
            _check_number_list(obj["normalized_center"], 2, f"{path}.normalized_center", errors)

        if "position_camera" in obj:
            _check_number_list(obj["position_camera"], 3, f"{path}.position_camera", errors)

        if "dimensions_m" in obj:
            _check_number_list(obj["dimensions_m"], 3, f"{path}.dimensions_m", errors)

        for key in ["bbox_area_pixels", "bbox_area_normalized", "distance_camera_m", "focus_distance"]:
            if key in obj and not _is_number(obj[key]):
                errors.append(f"{path}.{key} must be numeric")

        for key in ["is_in_front_of_camera", "is_focused_object"]:
            if key in obj and not isinstance(obj[key], bool):
                errors.append(f"{path}.{key} must be boolean")

        if obj.get("is_focused_object") is True:
            focused_count += 1

    if objects and focused_count != 1:
        errors.append(f"Expected exactly one focused object for non-empty frame, found {focused_count}")

    if is_null_sample and objects:
        errors.append("Null sample must not contain target objects")
    elif not objects and not is_null_sample:
        errors.append("Empty frame must be explicitly marked is_null_sample=true")

    return errors


def _records_from_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict) and "frames" in payload:
        frames = payload["frames"]

        if not isinstance(frames, list):
            raise ValueError("JSON field 'frames' must be a list")

        return frames

    if isinstance(payload, dict):
        return [payload]

    raise ValueError("JSON vision labels must be a record, list of records, or object with a 'frames' list")


def load_vision_label_file(path: str | Path, validate: bool = True) -> list[dict[str, Any]]:
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Vision label file does not exist: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported vision label extension: {path.suffix}")

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records: list[dict[str, Any]] = []

        with path.open("r") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc

                if validate:
                    errors = validate_vision_label_record(record)

                    if errors:
                        joined = "; ".join(errors)
                        raise ValueError(f"Invalid vision label in {path}:{line_number}: {joined}")

                records.append(record)

        return records

    with path.open("r") as f:
        records = _records_from_json_payload(json.load(f))

    if validate:
        for index, record in enumerate(records):
            errors = validate_vision_label_record(record)

            if errors:
                joined = "; ".join(errors)
                raise ValueError(f"Invalid vision label in {path} record {index}: {joined}")

    return records


def iter_vision_label_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []

    for item in paths:
        path = Path(item).expanduser().resolve()

        if path.is_dir():
            files.extend(
                sorted(
                    child for child in path.rglob("*")
                    if child.suffix.lower() in SUPPORTED_EXTENSIONS
                )
            )
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Vision label path does not exist: {path}")

    return files


def load_vision_labels(paths: Iterable[str | Path], validate: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for path in iter_vision_label_files(paths):
        records.extend(load_vision_label_file(path, validate=validate))

    return records


def summarize_vision_labels(records: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    semantic_class_counts: Counter[str] = Counter()
    focused_class_counts: Counter[str] = Counter()
    focused_semantic_class_counts: Counter[str] = Counter()

    num_objects = 0
    num_focused_objects = 0
    num_null_frames = 0
    distance_values: list[float] = []
    bbox_area_values: list[float] = []

    for record in records:
        if record.get("is_null_sample") is True:
            num_null_frames += 1

        for obj in record["objects"]:
            class_name = obj["class_name"]
            class_counts[class_name] += 1
            if obj.get("semantic_class"):
                semantic_class_counts[obj["semantic_class"]] += 1
            num_objects += 1

            distance_values.append(float(obj["distance_camera_m"]))
            bbox_area_values.append(float(obj["bbox_area_normalized"]))

            if obj["is_focused_object"]:
                focused_class_counts[class_name] += 1
                if obj.get("semantic_class"):
                    focused_semantic_class_counts[obj["semantic_class"]] += 1
                num_focused_objects += 1

    return {
        "num_frames": len(records),
        "num_objects": num_objects,
        "num_focused_objects": num_focused_objects,
        "num_null_frames": num_null_frames,
        "null_frame_fraction": num_null_frames / len(records) if records else 0.0,
        "class_counts": dict(class_counts),
        "semantic_class_counts": dict(semantic_class_counts),
        "focused_class_counts": dict(focused_class_counts),
        "focused_semantic_class_counts": dict(focused_semantic_class_counts),
        "mean_objects_per_frame": num_objects / len(records) if records else 0.0,
        "mean_distance_camera_m": sum(distance_values) / len(distance_values) if distance_values else 0.0,
        "mean_bbox_area_normalized": sum(bbox_area_values) / len(bbox_area_values) if bbox_area_values else 0.0,
    }
