from __future__ import annotations

from pathlib import Path
import csv
import json
import os
from typing import Any

import numpy as np
from PIL import Image

from .vision_label_dataset import load_vision_labels


DEPTH_ANNOTATION_TYPE = "type.unity.com/unity.solo.DepthAnnotation"
SEMANTIC_SEGMENTATION_ANNOTATION_TYPE = (
    "type.unity.com/unity.solo.SemanticSegmentationAnnotation"
)


def _clamp(value: float, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(round(value))))


def _safe_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _semantic_class(obj: dict[str, Any]) -> str:
    value = obj.get("semantic_class")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "Vision object is missing semantic_class. Run semantic enrichment or regenerate labels with the updated Unity exporter."
        )
    return value


def _crop_box(bbox_xyxy: list[float], image_width: int, image_height: int, padding: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy

    width = x2 - x1
    height = y2 - y1

    pad_x = width * padding
    pad_y = height * padding

    left = _clamp(x1 - pad_x, 0, image_width)
    top = _clamp(y1 - pad_y, 0, image_height)
    right = _clamp(x2 + pad_x, 0, image_width)
    bottom = _clamp(y2 + pad_y, 0, image_height)

    return left, top, right, bottom


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No crop rows to write.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _frame_data_path(dataset_root: Path, image_path: str) -> Path:
    relative_image_path = Path(image_path)
    image_name = relative_image_path.name

    if not image_name.endswith(".camera.png"):
        raise ValueError(f"Expected a .camera.png image path, got: {image_path}")

    frame_data_name = image_name.removesuffix(".camera.png") + ".frame_data.json"
    return dataset_root / relative_image_path.parent / frame_data_name


def _frame_capture(dataset_root: Path, image_path: str) -> tuple[Path, dict[str, Any]]:
    frame_data_path = _frame_data_path(dataset_root, image_path)

    if not frame_data_path.exists():
        raise FileNotFoundError(f"Perception frame data does not exist: {frame_data_path}")

    with frame_data_path.open("r") as file:
        frame_data = json.load(file)

    image_name = Path(image_path).name
    captures = frame_data.get("captures", [])

    for capture in captures:
        if capture.get("filename") == image_name:
            return frame_data_path, capture

    if len(captures) == 1:
        return frame_data_path, captures[0]

    raise ValueError(
        f"Could not identify camera capture for {image_name}: {frame_data_path}"
    )


def _find_annotation(
    frame_data_path: Path,
    capture: dict[str, Any],
    annotation_type: str,
) -> dict[str, Any]:
    for annotation in capture.get("annotations", []):
        if annotation.get("@type") == annotation_type:
            return annotation

    raise ValueError(
        f"Frame has no {annotation_type.rsplit('.', 1)[-1]}: {frame_data_path}"
    )


def _depth_annotation(dataset_root: Path, image_path: str) -> tuple[Path, str]:
    frame_data_path, capture = _frame_capture(dataset_root, image_path)
    annotation = _find_annotation(
        frame_data_path,
        capture,
        DEPTH_ANNOTATION_TYPE,
    )

    filename = annotation.get("filename")

    if not filename:
        raise ValueError(f"Depth annotation has no filename: {frame_data_path}")

    strategy = str(annotation.get("measurementStrategy", "Depth"))
    depth_path = frame_data_path.parent / filename

    if not depth_path.exists():
        raise FileNotFoundError(f"Depth image does not exist: {depth_path}")

    return depth_path, strategy


def _semantic_annotation(
    dataset_root: Path,
    image_path: str,
    semantic_class: str,
) -> tuple[Path, tuple[int, int, int, int]]:
    frame_data_path, capture = _frame_capture(dataset_root, image_path)
    annotation = _find_annotation(
        frame_data_path,
        capture,
        SEMANTIC_SEGMENTATION_ANNOTATION_TYPE,
    )

    filename = annotation.get("filename")
    if not filename:
        raise ValueError(
            f"Semantic segmentation annotation has no filename: {frame_data_path}"
        )

    segmentation_path = frame_data_path.parent / filename
    if not segmentation_path.exists():
        raise FileNotFoundError(
            f"Semantic segmentation image does not exist: {segmentation_path}"
        )

    matching_colors = [
        instance.get("pixelValue")
        for instance in annotation.get("instances", [])
        if instance.get("labelName") == semantic_class
    ]
    matching_colors = [
        color
        for color in matching_colors
        if isinstance(color, list) and len(color) >= 3
    ]

    if len(matching_colors) != 1:
        raise ValueError(
            f"Expected one semantic color for '{semantic_class}', found "
            f"{len(matching_colors)}: {frame_data_path}"
        )

    color = list(matching_colors[0])
    if len(color) == 3:
        color.append(255)

    return segmentation_path, tuple(int(value) for value in color[:4])


def _camera_calibration(
    dataset_root: Path,
    image_path: str,
    depth_size: tuple[int, int],
) -> dict[str, float]:
    frame_data_path, capture = _frame_capture(dataset_root, image_path)
    matrix = capture.get("matrix")
    dimensions = capture.get("dimension")
    rotation = capture.get("rotation")

    if not isinstance(matrix, list) or len(matrix) < 5:
        raise ValueError(f"Camera capture has no projection matrix: {frame_data_path}")
    if not isinstance(dimensions, list) or len(dimensions) < 2:
        raise ValueError(f"Camera capture has no image dimensions: {frame_data_path}")
    if not isinstance(rotation, list) or len(rotation) != 4:
        raise ValueError(f"Camera capture has no world rotation: {frame_data_path}")

    capture_width = float(dimensions[0])
    capture_height = float(dimensions[1])
    depth_width, depth_height = depth_size

    if capture_width <= 0.0 or capture_height <= 0.0:
        raise ValueError(f"Camera capture dimensions must be positive: {frame_data_path}")

    scale_x = depth_width / capture_width
    scale_y = depth_height / capture_height

    return {
        "depth_image_width": float(depth_width),
        "depth_image_height": float(depth_height),
        "camera_fx_px": float(matrix[0]) * capture_width * 0.5 * scale_x,
        "camera_fy_px": float(matrix[4]) * capture_height * 0.5 * scale_y,
        "camera_cx_px": capture_width * 0.5 * scale_x,
        "camera_cy_px": capture_height * 0.5 * scale_y,
        "camera_rotation_world_x": float(rotation[0]),
        "camera_rotation_world_y": float(rotation[1]),
        "camera_rotation_world_z": float(rotation[2]),
        "camera_rotation_world_w": float(rotation[3]),
    }


def _read_target_mask(
    segmentation_path: Path,
    pixel_value: tuple[int, int, int, int],
    bbox_xyxy: list[float],
) -> np.ndarray:
    with Image.open(segmentation_path) as image:
        rgba = np.asarray(image.convert("RGBA"))

    color = np.asarray(pixel_value, dtype=rgba.dtype)
    mask = np.all(rgba == color, axis=2)

    height, width = mask.shape
    x1, y1, x2, y2 = bbox_xyxy
    left = _clamp(x1, 0, width)
    top = _clamp(y1, 0, height)
    right = _clamp(x2, 0, width)
    bottom = _clamp(y2, 0, height)

    bounded_mask = np.zeros_like(mask)
    bounded_mask[top:bottom, left:right] = mask[top:bottom, left:right]

    if not np.any(bounded_mask):
        raise ValueError(
            f"Target mask contains no pixels for bbox {bbox_xyxy}: {segmentation_path}"
        )

    return bounded_mask


def _resize_binary_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = image.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def _read_metric_depth(depth_path: Path) -> np.ndarray:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

    import cv2

    encoded = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if encoded is None:
        raise ValueError(f"Could not read depth EXR: {depth_path}")

    if encoded.ndim == 2:
        depth = encoded
    elif encoded.ndim == 3 and encoded.shape[2] >= 3:
        # OpenCV loads color channels as BGR(A); Unity stores metric depth in R.
        depth = encoded[:, :, 2]
    else:
        raise ValueError(
            f"Unsupported depth image shape {encoded.shape}: {depth_path}"
        )

    return np.asarray(depth, dtype=np.float32)


def _depth_crop_box(
    rgb_crop_box: tuple[int, int, int, int],
    rgb_size: tuple[int, int],
    depth_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    left, top, right, bottom = rgb_crop_box
    rgb_width, rgb_height = rgb_size
    depth_width, depth_height = depth_size

    scale_x = depth_width / rgb_width
    scale_y = depth_height / rgb_height

    return (
        _clamp(left * scale_x, 0, depth_width),
        _clamp(top * scale_y, 0, depth_height),
        _clamp(right * scale_x, 0, depth_width),
        _clamp(bottom * scale_y, 0, depth_height),
    )


def export_vision_crops(
    vision_labels_path: str | Path,
    unity_image_root: str | Path,
    output_dir: str | Path,
    padding: float = 0.10,
    validate: bool = True,
    include_depth: bool = False,
    include_target_mask: bool = False,
) -> Path:
    vision_labels_path = Path(vision_labels_path).expanduser().resolve()
    unity_image_root = Path(unity_image_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    image_output_dir = output_dir / "images"
    depth_output_dir = output_dir / "depth"
    mask_output_dir = output_dir / "target_masks"
    masked_depth_output_dir = output_dir / "masked_depth"
    label_output_path = output_dir / "crop_labels.csv"
    frame_manifest_output_path = output_dir / "frame_manifest.csv"

    image_output_dir.mkdir(parents=True, exist_ok=True)

    if include_depth:
        depth_output_dir.mkdir(parents=True, exist_ok=True)
    if include_target_mask:
        mask_output_dir.mkdir(parents=True, exist_ok=True)
    if include_depth and include_target_mask:
        masked_depth_output_dir.mkdir(parents=True, exist_ok=True)

    records = load_vision_labels([vision_labels_path], validate=validate)

    rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []

    for frame_index, record in enumerate(records):
        source_image_path = unity_image_root / record["image_path"]

        if not source_image_path.exists():
            raise FileNotFoundError(f"Source image does not exist: {source_image_path}")

        frame_rows.append({
            "frame_id": record["frame_id"],
            "scene_id": record["scene_id"],
            "source_image_path": str(source_image_path),
            "is_null_sample": int(record.get("is_null_sample") is True),
            "num_target_objects": len(record["objects"]),
        })

        source_depth_path: Path | None = None
        depth_measurement_strategy = ""
        depth_image: np.ndarray | None = None
        calibration: dict[str, float] = {}

        if include_depth:
            source_depth_path, depth_measurement_strategy = _depth_annotation(
                unity_image_root,
                record["image_path"],
            )
            depth_image = _read_metric_depth(source_depth_path)
            try:
                calibration = _camera_calibration(
                    unity_image_root,
                    record["image_path"],
                    depth_size=(depth_image.shape[1], depth_image.shape[0]),
                )
            except ValueError:
                if include_target_mask:
                    raise

        with Image.open(source_image_path) as image:
            image = image.convert("RGB")
            image_width, image_height = image.size

            for object_index, obj in enumerate(record["objects"]):
                semantic_class = _semantic_class(obj)
                bbox_xyxy = obj["bbox_xyxy"]
                left, top, right, bottom = _crop_box(
                    bbox_xyxy=bbox_xyxy,
                    image_width=image_width,
                    image_height=image_height,
                    padding=padding,
                )

                if right <= left or bottom <= top:
                    continue

                crop = image.crop((left, top, right, bottom))

                crop_name = (
                    f"{_safe_name(record['frame_id'])}"
                    f"_obj{object_index:02d}_"
                    f"{_safe_name(semantic_class)}.png"
                )

                crop_path = image_output_dir / crop_name
                crop.save(crop_path)

                depth_crop_path = ""
                depth_valid_fraction = ""
                depth_median_m = ""
                depth_left = depth_top = depth_right = depth_bottom = ""
                source_segmentation_path = ""
                target_mask_path = ""
                masked_depth_crop_path = ""
                target_mask_fraction = ""
                target_depth_valid_fraction = ""
                target_depth_min_m = ""
                target_depth_p01_m = ""
                target_depth_median_m = ""
                target_depth_p99_m = ""
                target_depth_max_m = ""
                target_mask: np.ndarray | None = None

                if include_target_mask:
                    segmentation_path, pixel_value = _semantic_annotation(
                        unity_image_root,
                        record["image_path"],
                        semantic_class,
                    )
                    source_segmentation_path = str(segmentation_path)
                    full_target_mask = _read_target_mask(
                        segmentation_path=segmentation_path,
                        pixel_value=pixel_value,
                        bbox_xyxy=bbox_xyxy,
                    )

                    if full_target_mask.shape != (image_height, image_width):
                        raise ValueError(
                            "Semantic segmentation and RGB dimensions do not match: "
                            f"{segmentation_path} has {full_target_mask.shape[::-1]}, "
                            f"RGB has {(image_width, image_height)}."
                        )

                    target_mask = full_target_mask[top:bottom, left:right]
                    mask_file = mask_output_dir / crop_name
                    Image.fromarray(
                        target_mask.astype(np.uint8) * 255,
                        mode="L",
                    ).save(mask_file)
                    target_mask_path = str(mask_file)
                    target_mask_fraction = float(
                        np.count_nonzero(target_mask) / target_mask.size
                    )

                if depth_image is not None:
                    depth_height, depth_width = depth_image.shape
                    depth_left, depth_top, depth_right, depth_bottom = _depth_crop_box(
                        rgb_crop_box=(left, top, right, bottom),
                        rgb_size=(image_width, image_height),
                        depth_size=(depth_width, depth_height),
                    )

                    if depth_right <= depth_left or depth_bottom <= depth_top:
                        raise ValueError(
                            f"Depth crop is empty for {record['frame_id']} object {object_index}."
                        )

                    depth_crop = depth_image[
                        depth_top:depth_bottom,
                        depth_left:depth_right,
                    ]
                    depth_crop_file = depth_output_dir / crop_name.replace(".png", ".npy")
                    np.save(depth_crop_file, depth_crop.astype(np.float32, copy=False))
                    depth_crop_path = str(depth_crop_file)

                    valid_depth = depth_crop[
                        np.isfinite(depth_crop) & (depth_crop > 0.0)
                    ]
                    depth_valid_fraction = float(valid_depth.size / depth_crop.size)
                    depth_median_m = (
                        float(np.median(valid_depth))
                        if valid_depth.size
                        else 0.0
                    )

                    if include_target_mask:
                        depth_target_mask = _resize_binary_mask(
                            full_target_mask,
                            size=(depth_width, depth_height),
                        )[depth_top:depth_bottom, depth_left:depth_right]

                        if depth_target_mask.shape != depth_crop.shape:
                            raise ValueError(
                                "Target mask and depth crop dimensions do not match for "
                                f"{record['frame_id']} object {object_index}."
                            )

                        masked_depth = np.where(
                            depth_target_mask,
                            depth_crop,
                            0.0,
                        ).astype(np.float32, copy=False)
                        masked_depth_file = (
                            masked_depth_output_dir
                            / crop_name.replace(".png", ".npy")
                        )
                        np.save(masked_depth_file, masked_depth)
                        masked_depth_crop_path = str(masked_depth_file)

                        target_depth = masked_depth[
                            depth_target_mask
                            & np.isfinite(masked_depth)
                            & (masked_depth > 0.0)
                        ]
                        target_pixel_count = int(np.count_nonzero(depth_target_mask))
                        target_depth_valid_fraction = (
                            float(target_depth.size / target_pixel_count)
                            if target_pixel_count
                            else 0.0
                        )

                        if target_depth.size:
                            quantiles = np.quantile(
                                target_depth,
                                [0.01, 0.5, 0.99],
                            )
                            target_depth_min_m = float(np.min(target_depth))
                            target_depth_p01_m = float(quantiles[0])
                            target_depth_median_m = float(quantiles[1])
                            target_depth_p99_m = float(quantiles[2])
                            target_depth_max_m = float(np.max(target_depth))
                        else:
                            target_depth_min_m = 0.0
                            target_depth_p01_m = 0.0
                            target_depth_median_m = 0.0
                            target_depth_p99_m = 0.0
                            target_depth_max_m = 0.0

                position_camera = obj["position_camera"]
                dimensions = obj["dimensions_m"]
                bbox_width = max(0.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0]))
                bbox_height = max(0.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))

                row = {
                    "crop_image_path": str(crop_path),
                    "source_image_path": str(source_image_path),
                    "source_depth_path": str(source_depth_path) if source_depth_path else "",
                    "depth_crop_path": depth_crop_path,
                    "source_segmentation_path": source_segmentation_path,
                    "target_mask_path": target_mask_path,
                    "masked_depth_crop_path": masked_depth_crop_path,
                    "depth_measurement_strategy": depth_measurement_strategy,
                    "depth_valid_fraction": depth_valid_fraction,
                    "depth_median_m": depth_median_m,
                    "target_mask_fraction": target_mask_fraction,
                    "target_depth_valid_fraction": target_depth_valid_fraction,
                    "target_depth_min_m": target_depth_min_m,
                    "target_depth_p01_m": target_depth_p01_m,
                    "target_depth_median_m": target_depth_median_m,
                    "target_depth_p99_m": target_depth_p99_m,
                    "target_depth_max_m": target_depth_max_m,
                    "source_image_width": image_width,
                    "source_image_height": image_height,
                    "frame_id": record["frame_id"],
                    "scene_id": record["scene_id"],

                    "object_index": object_index,
                    "object_id": obj["object_id"],
                    "object_name": obj["object_name"],
                    "class_name": obj["class_name"],
                    "semantic_class": semantic_class,
                    "is_focused_object": int(obj["is_focused_object"]),

                    "bbox_x1": bbox_xyxy[0],
                    "bbox_y1": bbox_xyxy[1],
                    "bbox_x2": bbox_xyxy[2],
                    "bbox_y2": bbox_xyxy[3],
                    "bbox_width_norm": bbox_width / image_width,
                    "bbox_height_norm": bbox_height / image_height,
                    "bbox_center_x_norm": (float(bbox_xyxy[0]) + bbox_width / 2.0) / image_width,
                    "bbox_center_y_norm": (float(bbox_xyxy[1]) + bbox_height / 2.0) / image_height,
                    "bbox_aspect_ratio": bbox_width / bbox_height if bbox_height > 0.0 else 0.0,
                    "crop_left": left,
                    "crop_top": top,
                    "crop_right": right,
                    "crop_bottom": bottom,
                    "crop_width": right - left,
                    "crop_height": bottom - top,
                    "depth_crop_left": depth_left,
                    "depth_crop_top": depth_top,
                    "depth_crop_right": depth_right,
                    "depth_crop_bottom": depth_bottom,

                    "bbox_area_normalized": obj["bbox_area_normalized"],
                    "distance_camera_m": obj["distance_camera_m"],

                    "position_camera_x": position_camera[0],
                    "position_camera_y": position_camera[1],
                    "position_camera_z": position_camera[2],

                    "dimension_x": dimensions[0],
                    "dimension_y": dimensions[1],
                    "dimension_z": dimensions[2],
                }

                row.update(calibration)
                rows.append(row)

    _write_csv(rows, label_output_path)
    _write_csv(frame_rows, frame_manifest_output_path)

    return label_output_path
