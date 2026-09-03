from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any

from .vision_label_dataset import load_vision_label_file, validate_vision_label_record


BOUNDING_BOX_ANNOTATION_TYPE = "type.unity.com/unity.solo.BoundingBox2DAnnotation"


def _bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection

    return intersection / union if union > 0.0 else 0.0


def _object_identity(obj: dict[str, Any]) -> str:
    object_name = str(obj.get("object_name", "")).strip()
    return re.sub(r"^SampleObject_\d+_", "", object_name)


def _frame_data_path(dataset_root: Path, image_path: str) -> Path:
    relative_image_path = Path(image_path)
    image_name = relative_image_path.name

    if not image_name.endswith(".camera.png"):
        raise ValueError(f"Expected a .camera.png image path, got: {image_path}")

    frame_data_name = image_name.removesuffix(".camera.png") + ".frame_data.json"
    return dataset_root / relative_image_path.parent / frame_data_name


def _perception_boxes(frame_data_path: Path) -> list[dict[str, Any]]:
    if not frame_data_path.exists():
        raise FileNotFoundError(f"Perception frame data does not exist: {frame_data_path}")

    with frame_data_path.open("r") as f:
        frame_data = json.load(f)

    output: list[dict[str, Any]] = []

    for capture in frame_data.get("captures", []):
        for annotation in capture.get("annotations", []):
            if annotation.get("@type") != BOUNDING_BOX_ANNOTATION_TYPE:
                continue

            for value in annotation.get("values", []):
                origin = value.get("origin", [])
                dimensions = value.get("dimension", [])
                semantic_class = value.get("labelName")

                if len(origin) != 2 or len(dimensions) != 2 or not semantic_class:
                    continue

                x, y = (float(item) for item in origin)
                width, height = (float(item) for item in dimensions)

                output.append({
                    "semantic_class": str(semantic_class),
                    "bbox_xyxy": [x, y, x + width, y + height],
                })

    return output


def _match_objects(
    objects: list[dict[str, Any]],
    perception_boxes: list[dict[str, Any]],
    minimum_iou: float,
) -> dict[int, int]:
    if len(objects) == 1 and len(perception_boxes) == 1:
        return {0: 0}

    candidates: list[tuple[float, int, int]] = []

    for object_index, obj in enumerate(objects):
        for box_index, perception_box in enumerate(perception_boxes):
            overlap = _bbox_iou(obj["bbox_xyxy"], perception_box["bbox_xyxy"])
            candidates.append((overlap, object_index, box_index))

    matches: dict[int, int] = {}
    used_boxes: set[int] = set()

    for overlap, object_index, box_index in sorted(candidates, reverse=True):
        if overlap < minimum_iou:
            break
        if object_index in matches or box_index in used_boxes:
            continue

        matches[object_index] = box_index
        used_boxes.add(box_index)

    return matches


def enrich_vision_labels_with_semantic_classes(
    vision_labels_path: str | Path,
    dataset_root: str | Path,
    output_path: str | Path,
    minimum_iou: float = 0.05,
) -> dict[str, Any]:
    vision_labels_path = Path(vision_labels_path).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if output_path == vision_labels_path:
        raise ValueError("Semantic enrichment output must not overwrite the original vision labels.")
    if not 0.0 <= minimum_iou <= 1.0:
        raise ValueError("minimum_iou must be between 0 and 1.")

    records = load_vision_label_file(vision_labels_path, validate=True)
    identity_labels: dict[str, Counter[str]] = defaultdict(Counter)
    unmatched_objects: list[tuple[str, dict[str, Any]]] = []
    semantic_counts: Counter[str] = Counter()
    directly_matched = 0

    for record in records:
        frame_data_path = _frame_data_path(dataset_root, record["image_path"])
        perception_boxes = _perception_boxes(frame_data_path)
        objects = record["objects"]
        matches = _match_objects(objects, perception_boxes, minimum_iou=minimum_iou)

        for object_index, obj in enumerate(objects):
            if object_index not in matches:
                unmatched_objects.append((record["frame_id"], obj))
                continue

            semantic_class = perception_boxes[matches[object_index]]["semantic_class"]
            obj["semantic_class"] = semantic_class
            identity_labels[_object_identity(obj)][semantic_class] += 1
            semantic_counts[semantic_class] += 1
            directly_matched += 1

    identity_fallback_count = 0
    unresolved: list[dict[str, str]] = []

    for frame_id, obj in unmatched_objects:
        identity = _object_identity(obj)
        candidates = identity_labels.get(identity, Counter())

        if not candidates:
            unresolved.append({
                "frame_id": frame_id,
                "object_name": str(obj.get("object_name", "")),
                "reason": "no matched example for this object identity",
            })
            continue

        semantic_class, count = candidates.most_common(1)[0]
        competing_count = sum(candidates.values()) - count

        if competing_count > 0:
            unresolved.append({
                "frame_id": frame_id,
                "object_name": str(obj.get("object_name", "")),
                "reason": f"ambiguous identity labels: {dict(candidates)}",
            })
            continue

        obj["semantic_class"] = semantic_class
        semantic_counts[semantic_class] += 1
        identity_fallback_count += 1

    if unresolved:
        preview = "; ".join(
            f"{item['frame_id']} {item['object_name']}: {item['reason']}"
            for item in unresolved[:10]
        )
        raise ValueError(
            f"Could not assign semantic classes to {len(unresolved)} object(s). {preview}"
        )

    for record_index, record in enumerate(records):
        errors = validate_vision_label_record(record, require_semantic_class=True)
        if errors:
            raise ValueError(
                f"Enriched record {record_index} is invalid: {'; '.join(errors)}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    summary = {
        "input_path": str(vision_labels_path),
        "dataset_root": str(dataset_root),
        "output_path": str(output_path),
        "num_frames": len(records),
        "num_objects": sum(len(record["objects"]) for record in records),
        "direct_bbox_matches": directly_matched,
        "identity_fallback_matches": identity_fallback_count,
        "semantic_class_counts": dict(sorted(semantic_counts.items())),
        "minimum_iou": minimum_iou,
    }

    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    summary["summary_path"] = str(summary_path)
    return summary
