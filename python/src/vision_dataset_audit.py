from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .vision_label_dataset import load_vision_label_file, validate_vision_label_record


BOUNDING_BOX_ANNOTATION_TYPE = "type.unity.com/unity.solo.BoundingBox2DAnnotation"
SEMANTIC_SEGMENTATION_ANNOTATION_TYPE = (
    "type.unity.com/unity.solo.SemanticSegmentationAnnotation"
)
DEPTH_ANNOTATION_TYPE = "type.unity.com/unity.solo.DepthAnnotation"
RENDERED_OBJECT_INFO_METRIC_TYPE = "type.unity.com/unity.solo.RenderedObjectInfoMetric"


def _frame_data_path(dataset_root: Path, image_path: str) -> Path:
    relative_image_path = Path(image_path)
    image_name = relative_image_path.name

    if not image_name.endswith(".camera.png"):
        raise ValueError(f"Expected a .camera.png image path, got: {image_path}")

    frame_data_name = image_name.removesuffix(".camera.png") + ".frame_data.json"
    return dataset_root / relative_image_path.parent / frame_data_name


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


def _perception_boxes(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []

    for annotation in annotations:
        if annotation.get("@type") != BOUNDING_BOX_ANNOTATION_TYPE:
            continue

        for value in annotation.get("values", []):
            origin = value.get("origin", [])
            dimensions = value.get("dimension", [])

            if len(origin) != 2 or len(dimensions) != 2:
                continue

            x, y = (float(item) for item in origin)
            width, height = (float(item) for item in dimensions)
            boxes.append({
                "bbox_xyxy": [x, y, x + width, y + height],
                "semantic_class": str(value.get("labelName", "")).strip(),
            })

    return boxes


def _match_objects(
    objects: list[dict[str, Any]],
    boxes: list[dict[str, Any]],
    minimum_iou: float,
) -> dict[int, int]:
    candidates: list[tuple[float, int, int]] = []

    for object_index, obj in enumerate(objects):
        bbox = obj.get("bbox_xyxy")

        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        for box_index, box in enumerate(boxes):
            overlap = _bbox_iou(
                [float(value) for value in bbox],
                box["bbox_xyxy"],
            )
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


def _visible_semantic_pixels(path: Path) -> int:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    return int(np.count_nonzero(np.any(rgb != 0, axis=2)))


def _add_issue(
    issues: list[dict[str, str]],
    code: str,
    message: str,
) -> None:
    issues.append({"code": code, "message": message})


def audit_and_filter_vision_dataset(
    vision_labels_path: str | Path,
    dataset_root: str | Path,
    filtered_output_path: str | Path,
    report_output_path: str | Path,
    minimum_iou: float = 0.05,
    minimum_visible_pixels: int = 1,
    require_depth: bool = False,
) -> dict[str, Any]:
    if not 0.0 <= minimum_iou <= 1.0:
        raise ValueError("minimum_iou must be between 0 and 1.")
    if minimum_visible_pixels < 1:
        raise ValueError("minimum_visible_pixels must be at least 1.")

    vision_labels_path = Path(vision_labels_path).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    filtered_output_path = Path(filtered_output_path).expanduser().resolve()
    report_output_path = Path(report_output_path).expanduser().resolve()

    if filtered_output_path == vision_labels_path:
        raise ValueError("Filtered output must not overwrite the raw vision labels.")

    records = load_vision_label_file(vision_labels_path, validate=False)
    valid_records: list[dict[str, Any]] = []
    invalid_frames: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    input_class_counts: Counter[str] = Counter()
    valid_class_counts: Counter[str] = Counter()
    seen_frame_ids: set[str] = set()
    seen_image_paths: set[str] = set()

    for record_index, record in enumerate(records):
        frame_id = str(record.get("frame_id", f"record_{record_index}"))
        image_path = str(record.get("image_path", ""))
        objects = record.get("objects", [])
        is_null_sample = record.get("is_null_sample") is True
        issues: list[dict[str, str]] = []

        if isinstance(objects, list):
            input_class_counts.update(
                str(obj.get("semantic_class", "unknown"))
                for obj in objects
                if isinstance(obj, dict)
            )

        schema_errors = validate_vision_label_record(
            record,
            require_semantic_class=True,
        )
        for error in schema_errors:
            _add_issue(issues, "invalid_rich_label_schema", error)

        if frame_id in seen_frame_ids:
            _add_issue(issues, "duplicate_frame_id", f"Duplicate frame_id: {frame_id}")
        seen_frame_ids.add(frame_id)

        if image_path in seen_image_paths:
            _add_issue(issues, "duplicate_image_path", f"Duplicate image_path: {image_path}")
        seen_image_paths.add(image_path)

        source_image_path = dataset_root / image_path
        if not source_image_path.exists():
            _add_issue(
                issues,
                "missing_rgb_image",
                f"RGB image does not exist: {source_image_path}",
            )

        frame_data: dict[str, Any] | None = None
        frame_data_path: Path | None = None

        try:
            frame_data_path = _frame_data_path(dataset_root, image_path)
        except ValueError as exc:
            _add_issue(issues, "invalid_image_path", str(exc))

        if frame_data_path is not None:
            if not frame_data_path.exists():
                _add_issue(
                    issues,
                    "missing_frame_data",
                    f"Frame data does not exist: {frame_data_path}",
                )
            else:
                try:
                    with frame_data_path.open("r") as file:
                        frame_data = json.load(file)
                except (json.JSONDecodeError, OSError) as exc:
                    _add_issue(
                        issues,
                        "invalid_frame_data",
                        f"Could not read frame data: {exc}",
                    )

        visible_pixels: int | None = None

        if frame_data is not None and frame_data_path is not None:
            annotations = [
                annotation
                for capture in frame_data.get("captures", [])
                for annotation in capture.get("annotations", [])
            ]
            boxes = _perception_boxes(annotations)
            object_list = objects if isinstance(objects, list) else []

            if len(boxes) != len(object_list):
                _add_issue(
                    issues,
                    "bbox_count_mismatch",
                    f"Rich objects={len(object_list)}, Perception bboxes={len(boxes)}.",
                )
            else:
                matches = _match_objects(object_list, boxes, minimum_iou)

                if len(matches) != len(object_list):
                    _add_issue(
                        issues,
                        "bbox_match_failure",
                        f"Matched {len(matches)}/{len(object_list)} objects at IoU >= {minimum_iou}.",
                    )
                else:
                    for object_index, box_index in matches.items():
                        rich_class = str(
                            object_list[object_index].get("semantic_class", "")
                        ).strip()
                        perception_class = boxes[box_index]["semantic_class"]

                        if rich_class != perception_class:
                            _add_issue(
                                issues,
                                "semantic_class_mismatch",
                                f"Object {object_index}: rich='{rich_class}', Perception='{perception_class}'.",
                            )

            segmentation_annotations = [
                annotation
                for annotation in annotations
                if annotation.get("@type") == SEMANTIC_SEGMENTATION_ANNOTATION_TYPE
            ]

            if len(segmentation_annotations) != 1:
                _add_issue(
                    issues,
                    "segmentation_annotation_count",
                    f"Expected one segmentation annotation, found {len(segmentation_annotations)}.",
                )
            else:
                segmentation_filename = segmentation_annotations[0].get("filename", "")
                segmentation_path = frame_data_path.parent / segmentation_filename

                if not segmentation_path.exists():
                    _add_issue(
                        issues,
                        "missing_segmentation_image",
                        f"Segmentation image does not exist: {segmentation_path}",
                    )
                else:
                    visible_pixels = _visible_semantic_pixels(segmentation_path)

                    if is_null_sample and visible_pixels != 0:
                        _add_issue(
                            issues,
                            "unexpected_null_sample_pixels",
                            f"Null sample contains {visible_pixels} semantic pixels.",
                        )
                    elif not is_null_sample and visible_pixels < minimum_visible_pixels:
                        _add_issue(
                            issues,
                            "insufficient_visible_pixels",
                            f"Visible semantic pixels={visible_pixels}, minimum={minimum_visible_pixels}.",
                        )

            rendered_values = [
                value
                for metric in frame_data.get("metrics", [])
                if metric.get("@type") == RENDERED_OBJECT_INFO_METRIC_TYPE
                for value in metric.get("values", [])
            ]

            if len(rendered_values) != len(object_list):
                _add_issue(
                    issues,
                    "rendered_object_count_mismatch",
                    f"Rich objects={len(object_list)}, rendered objects={len(rendered_values)}.",
                )

            if require_depth:
                depth_annotations = [
                    annotation
                    for annotation in annotations
                    if annotation.get("@type") == DEPTH_ANNOTATION_TYPE
                ]

                if len(depth_annotations) != 1:
                    _add_issue(
                        issues,
                        "depth_annotation_count",
                        f"Expected one depth annotation, found {len(depth_annotations)}.",
                    )
                else:
                    depth_filename = depth_annotations[0].get("filename", "")
                    depth_path = frame_data_path.parent / depth_filename

                    if not depth_path.exists():
                        _add_issue(
                            issues,
                            "missing_depth_image",
                            f"Depth image does not exist: {depth_path}",
                        )

        if issues:
            issue_counts.update(issue["code"] for issue in issues)
            invalid_frames.append({
                "record_index": record_index,
                "frame_id": frame_id,
                "image_path": image_path,
                "visible_semantic_pixels": visible_pixels,
                "issues": issues,
            })
        else:
            valid_records.append(record)
            valid_class_counts.update(
                str(obj["semantic_class"])
                for obj in objects
            )

    filtered_output_path.parent.mkdir(parents=True, exist_ok=True)
    with filtered_output_path.open("w") as file:
        for record in valid_records:
            file.write(json.dumps(record, separators=(",", ":")) + "\n")

    report = {
        "input_path": str(vision_labels_path),
        "dataset_root": str(dataset_root),
        "filtered_output_path": str(filtered_output_path),
        "report_output_path": str(report_output_path),
        "settings": {
            "minimum_iou": minimum_iou,
            "minimum_visible_pixels": minimum_visible_pixels,
            "require_depth": require_depth,
        },
        "num_input_frames": len(records),
        "num_valid_frames": len(valid_records),
        "num_invalid_frames": len(invalid_frames),
        "num_input_objects": sum(
            len(record.get("objects", []))
            for record in records
            if isinstance(record.get("objects", []), list)
        ),
        "num_valid_objects": sum(len(record["objects"]) for record in valid_records),
        "num_input_null_frames": sum(
            record.get("is_null_sample") is True
            for record in records
        ),
        "num_valid_null_frames": sum(
            record.get("is_null_sample") is True
            for record in valid_records
        ),
        "input_semantic_class_counts": dict(sorted(input_class_counts.items())),
        "valid_semantic_class_counts": dict(sorted(valid_class_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "invalid_frames": invalid_frames,
    }

    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    with report_output_path.open("w") as file:
        json.dump(report, file, indent=2)

    return report
