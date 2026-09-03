from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json
import math
from typing import Any

import numpy as np


POSITION_AXES = ("x", "y", "z")
DIMENSION_AXES = ("x", "y", "z")


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Crop row is missing required value: {key}")
    return float(value)


def _quaternion_rotation_matrix(
    x: float,
    y: float,
    z: float,
    w: float,
) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("Camera rotation quaternion has zero length.")

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def backproject_masked_depth(
    masked_depth: np.ndarray,
    crop_left: float,
    crop_top: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    measurement_strategy: str,
) -> np.ndarray:
    depth = np.asarray(masked_depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D masked depth array, got {depth.shape}.")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("Camera focal lengths must be positive.")

    valid = np.isfinite(depth) & (depth > 0.0)
    rows, columns = np.nonzero(valid)
    if rows.size == 0:
        raise ValueError("Masked depth crop contains no valid target pixels.")

    ranges = depth[valid]
    image_x = crop_left + columns.astype(np.float64) + 0.5
    image_y = crop_top + rows.astype(np.float64) + 0.5
    rays = np.stack([
        (image_x - cx) / fx,
        -(image_y - cy) / fy,
        np.ones_like(image_x),
    ], axis=1)

    strategy = measurement_strategy.strip().lower()
    if strategy in {"depth", "distance", "range"}:
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    elif strategy not in {"z", "zdepth", "camera_z"}:
        raise ValueError(
            f"Unsupported depth measurement strategy: {measurement_strategy}"
        )

    return rays * ranges[:, None]


def estimate_depth_geometry(
    row: dict[str, Any],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> dict[str, Any]:
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Geometry quantiles must satisfy 0 <= lower < upper <= 1.")

    masked_depth_value = str(row.get("masked_depth_crop_path", "")).strip()
    if not masked_depth_value:
        raise ValueError("Crop row is missing required value: masked_depth_crop_path")

    masked_depth_path = Path(masked_depth_value).expanduser()
    if not masked_depth_path.exists():
        raise FileNotFoundError(
            f"Masked depth crop does not exist: {masked_depth_path}"
        )

    masked_depth = np.load(masked_depth_path.resolve(), allow_pickle=False)
    points_camera = backproject_masked_depth(
        masked_depth=masked_depth,
        crop_left=_float(row, "depth_crop_left"),
        crop_top=_float(row, "depth_crop_top"),
        fx=_float(row, "camera_fx_px"),
        fy=_float(row, "camera_fy_px"),
        cx=_float(row, "camera_cx_px"),
        cy=_float(row, "camera_cy_px"),
        measurement_strategy=str(row.get("depth_measurement_strategy", "Depth")),
    )

    raw_lower_camera = np.min(points_camera, axis=0)
    raw_upper_camera = np.max(points_camera, axis=0)
    robust_lower_camera = np.quantile(points_camera, lower_quantile, axis=0)
    robust_upper_camera = np.quantile(points_camera, upper_quantile, axis=0)

    raw_center_camera = (raw_lower_camera + raw_upper_camera) * 0.5
    robust_center_camera = (robust_lower_camera + robust_upper_camera) * 0.5

    rotation = _quaternion_rotation_matrix(
        _float(row, "camera_rotation_world_x"),
        _float(row, "camera_rotation_world_y"),
        _float(row, "camera_rotation_world_z"),
        _float(row, "camera_rotation_world_w"),
    )
    points_world_axes = points_camera @ rotation.T

    raw_dimensions_world = (
        np.max(points_world_axes, axis=0) - np.min(points_world_axes, axis=0)
    )
    robust_dimensions_world = (
        np.quantile(points_world_axes, upper_quantile, axis=0)
        - np.quantile(points_world_axes, lower_quantile, axis=0)
    )

    valid_depth = masked_depth[
        np.isfinite(masked_depth) & (masked_depth > 0.0)
    ]

    return {
        "num_target_depth_pixels": int(points_camera.shape[0]),
        "raw_center_camera": raw_center_camera,
        "robust_center_camera": robust_center_camera,
        "raw_distance_camera_m": float(np.linalg.norm(raw_center_camera)),
        "robust_distance_camera_m": float(np.linalg.norm(robust_center_camera)),
        "raw_dimensions_world": raw_dimensions_world,
        "robust_dimensions_world": robust_dimensions_world,
        "surface_depth_min_m": float(np.min(valid_depth)),
        "surface_depth_max_m": float(np.max(valid_depth)),
        "surface_depth_p_lower_m": float(np.quantile(valid_depth, lower_quantile)),
        "surface_depth_p_upper_m": float(np.quantile(valid_depth, upper_quantile)),
    }


def _prediction_row(
    row: dict[str, Any],
    estimate: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "frame_id": row.get("frame_id", ""),
        "scene_id": row.get("scene_id", ""),
        "object_name": row.get("object_name", ""),
        "semantic_class": row.get("semantic_class", ""),
        "crop_image_path": row.get("crop_image_path", ""),
        "target_mask_path": row.get("target_mask_path", ""),
        "masked_depth_crop_path": row.get("masked_depth_crop_path", ""),
        "num_target_depth_pixels": estimate["num_target_depth_pixels"],
        "target_mask_fraction": row.get("target_mask_fraction", ""),
        "target_depth_valid_fraction": row.get("target_depth_valid_fraction", ""),
        "surface_depth_min_m": estimate["surface_depth_min_m"],
        "surface_depth_max_m": estimate["surface_depth_max_m"],
        "surface_depth_p_lower_m": estimate["surface_depth_p_lower_m"],
        "surface_depth_p_upper_m": estimate["surface_depth_p_upper_m"],
        "bbox_width_norm": row.get("bbox_width_norm", ""),
        "bbox_height_norm": row.get("bbox_height_norm", ""),
        "bbox_center_x_norm": row.get("bbox_center_x_norm", ""),
        "bbox_center_y_norm": row.get("bbox_center_y_norm", ""),
        "bbox_area_normalized": row.get("bbox_area_normalized", ""),
        "bbox_aspect_ratio": row.get("bbox_aspect_ratio", ""),
    }

    true_distance = _float(row, "distance_camera_m")
    output["true_distance_camera_m"] = true_distance

    for method in ("raw", "robust"):
        predicted_distance = float(estimate[f"{method}_distance_camera_m"])
        output[f"{method}_distance_camera_m"] = predicted_distance
        output[f"{method}_abs_error_distance_camera_m"] = abs(
            predicted_distance - true_distance
        )

        center = estimate[f"{method}_center_camera"]
        dimensions = estimate[f"{method}_dimensions_world"]

        for index, axis in enumerate(POSITION_AXES):
            true_value = _float(row, f"position_camera_{axis}")
            predicted_value = float(center[index])
            output[f"true_position_camera_{axis}"] = true_value
            output[f"{method}_position_camera_{axis}"] = predicted_value
            output[f"{method}_abs_error_position_camera_{axis}"] = abs(
                predicted_value - true_value
            )

        for index, axis in enumerate(DIMENSION_AXES):
            true_value = _float(row, f"dimension_{axis}")
            predicted_value = float(dimensions[index])
            output[f"true_dimension_{axis}"] = true_value
            output[f"{method}_dimension_{axis}"] = predicted_value
            output[f"{method}_abs_error_dimension_{axis}"] = abs(
                predicted_value - true_value
            )

    return output


def _method_metrics(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    distance_errors = [
        float(row[f"{method}_abs_error_distance_camera_m"])
        for row in rows
    ]
    position_axis_errors = {
        axis: [
            float(row[f"{method}_abs_error_position_camera_{axis}"])
            for row in rows
        ]
        for axis in POSITION_AXES
    }
    dimension_axis_errors = {
        axis: [
            float(row[f"{method}_abs_error_dimension_{axis}"])
            for row in rows
        ]
        for axis in DIMENSION_AXES
    }

    position_euclidean = [
        math.sqrt(sum(position_axis_errors[axis][index] ** 2 for axis in POSITION_AXES))
        for index in range(len(rows))
    ]
    dimension_euclidean = [
        math.sqrt(sum(dimension_axis_errors[axis][index] ** 2 for axis in DIMENSION_AXES))
        for index in range(len(rows))
    ]

    return {
        "num_rows": len(rows),
        "mae_distance_camera_m": float(np.mean(distance_errors)),
        "median_abs_error_distance_camera_m": float(np.median(distance_errors)),
        "p90_abs_error_distance_camera_m": float(np.quantile(distance_errors, 0.90)),
        "max_abs_error_distance_camera_m": float(np.max(distance_errors)),
        **{
            f"mae_position_camera_{axis}": float(np.mean(values))
            for axis, values in position_axis_errors.items()
        },
        "mean_position_euclidean_error_m": float(np.mean(position_euclidean)),
        **{
            f"mae_dimension_{axis}": float(np.mean(values))
            for axis, values in dimension_axis_errors.items()
        },
        "mean_dimension_euclidean_error_m": float(np.mean(dimension_euclidean)),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_depth_geometry(
    crop_labels_csv: str | Path,
    predictions_output: str | Path,
    report_output: str | Path,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> dict[str, Any]:
    crop_labels_csv = Path(crop_labels_csv).expanduser().resolve()
    predictions_output = Path(predictions_output).expanduser().resolve()
    report_output = Path(report_output).expanduser().resolve()

    with crop_labels_csv.open(newline="") as file:
        crop_rows = list(csv.DictReader(file))

    predictions: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for index, row in enumerate(crop_rows):
        try:
            estimate = estimate_depth_geometry(
                row,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            )
            predictions.append(_prediction_row(row, estimate))
        except (FileNotFoundError, ValueError) as exc:
            skipped_rows.append({
                "row_index": index,
                "frame_id": row.get("frame_id", ""),
                "reason": str(exc),
            })

    if not predictions:
        raise ValueError("No valid masked-depth rows were available for geometry evaluation.")

    _write_csv(predictions, predictions_output)

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_class[str(prediction.get("semantic_class", "unknown"))].append(prediction)

    raw_metrics = _method_metrics(predictions, "raw")
    robust_metrics = _method_metrics(predictions, "robust")
    report = {
        "model_family": "calibrated_depth_geometry",
        "architecture": "masked_depth_geometry",
        "crop_labels_path": str(crop_labels_csv),
        "predictions_path": str(predictions_output),
        "num_input_rows": len(crop_rows),
        "num_evaluated_rows": len(predictions),
        "num_skipped_rows": len(skipped_rows),
        "skipped_rows": skipped_rows,
        "depth_measurement": (
            "Metric radial range is back-projected through the camera projection matrix."
        ),
        "raw_geometry_definition": (
            "Centre and extents of the complete visible target point-cloud min/max bounds; "
            "an upper bound for exact synthetic depth and masks."
        ),
        "robust_geometry_definition": (
            f"Centre and extents between point-cloud quantiles {lower_quantile:.3f} "
            f"and {upper_quantile:.3f}; less sensitive to real depth noise."
        ),
        "dimension_target_definition": "Unity world-axis-aligned renderer bounds.",
        "dimension_estimate_limitation": (
            "Visible point-cloud extents cannot recover fully occluded object surfaces."
        ),
        "test_metrics": raw_metrics,
        "raw_geometry": raw_metrics,
        "robust_geometry": robust_metrics,
        "per_class": {
            class_name: {
                "raw_geometry": _method_metrics(class_rows, "raw"),
                "robust_geometry": _method_metrics(class_rows, "robust"),
            }
            for class_name, class_rows in sorted(by_class.items())
        },
    }

    report_output.parent.mkdir(parents=True, exist_ok=True)
    with report_output.open("w") as file:
        json.dump(report, file, indent=2)

    return report
