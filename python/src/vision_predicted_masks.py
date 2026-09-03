from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Crop label CSV does not exist: {resolved}")
    with resolved.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Crop label CSV contains no rows: {resolved}")

    required = {
        "source_image_path",
        "depth_crop_path",
        "semantic_class",
        "frame_id",
        "object_index",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "crop_left",
        "crop_top",
        "crop_right",
        "crop_bottom",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Crop label CSV is missing required columns: {missing}")
    return rows


def bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def select_prediction_index(
    ground_truth_bbox: np.ndarray,
    ground_truth_class_id: int,
    predicted_boxes: np.ndarray,
    predicted_class_ids: np.ndarray,
    used_indices: set[int] | None = None,
) -> tuple[int | None, float]:
    used = used_indices or set()
    candidates = [
        (index, bbox_iou(ground_truth_bbox, predicted_boxes[index]))
        for index in range(len(predicted_boxes))
        if index not in used and int(predicted_class_ids[index]) == ground_truth_class_id
    ]
    if not candidates:
        return None, 0.0
    return max(candidates, key=lambda item: item[1])


def rasterize_polygon_crop(
    polygon_xy: np.ndarray,
    crop_box: tuple[int, int, int, int],
) -> np.ndarray:
    left, top, right, bottom = crop_box
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid crop box: {crop_box}")

    polygon = np.asarray(polygon_xy, dtype=np.float64).reshape(-1, 2).copy()
    if len(polygon) < 3:
        raise ValueError("Predicted mask polygon has fewer than three points.")
    polygon[:, 0] -= left
    polygon[:, 1] -= top
    polygon = np.rint(polygon).astype(np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], color=1)
    return mask > 0


def rasterize_bbox_crop(
    bbox_xyxy: np.ndarray,
    crop_box: tuple[int, int, int, int],
) -> np.ndarray:
    left, top, right, bottom = crop_box
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid crop box: {crop_box}")

    bbox = np.asarray(bbox_xyxy, dtype=np.float64).reshape(4)
    box_left = max(0, int(np.floor(bbox[0] - left)))
    box_top = max(0, int(np.floor(bbox[1] - top)))
    box_right = min(width, int(np.ceil(bbox[2] - left)))
    box_bottom = min(height, int(np.ceil(bbox[3] - top)))
    if box_right <= box_left or box_bottom <= box_top:
        raise ValueError("Predicted bounding box does not overlap the evaluation crop.")

    mask = np.zeros((height, width), dtype=bool)
    mask[box_top:box_bottom, box_left:box_right] = True
    return mask


def _target_depth_statistics(masked_depth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    values = masked_depth[mask & np.isfinite(masked_depth) & (masked_depth > 0.0)]
    if values.size == 0:
        return {
            "target_depth_valid_fraction": 0.0,
            "target_depth_min_m": 0.0,
            "target_depth_p01_m": 0.0,
            "target_depth_median_m": 0.0,
            "target_depth_p99_m": 0.0,
            "target_depth_max_m": 0.0,
        }
    return {
        "target_depth_valid_fraction": float(values.size / max(1, np.count_nonzero(mask))),
        "target_depth_min_m": float(np.min(values)),
        "target_depth_p01_m": float(np.quantile(values, 0.01)),
        "target_depth_median_m": float(np.median(values)),
        "target_depth_p99_m": float(np.quantile(values, 0.99)),
        "target_depth_max_m": float(np.max(values)),
    }


def filter_predicted_mask_depth(
    depth: np.ndarray,
    predicted_mask: np.ndarray,
    mad_scale: float = 3.0,
    minimum_band_m: float = 0.08,
    maximum_band_m: float = 0.25,
) -> tuple[np.ndarray, dict[str, float]]:
    if mad_scale <= 0.0:
        raise ValueError("Depth MAD scale must be positive.")
    if not 0.0 < minimum_band_m <= maximum_band_m:
        raise ValueError("Depth band limits must satisfy 0 < minimum <= maximum.")

    valid = predicted_mask & np.isfinite(depth) & (depth > 0.0)
    values = depth[valid]
    if values.size == 0:
        raise ValueError("Predicted mask contains no valid depth pixels.")

    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    robust_sigma = 1.4826 * mad
    band = float(np.clip(mad_scale * robust_sigma, minimum_band_m, maximum_band_m))
    support = valid & (np.abs(depth - center) <= band)
    if not np.any(support):
        raise ValueError("Depth support filter removed every predicted-mask pixel.")

    return support, {
        "depth_filter_center_m": center,
        "depth_filter_mad_m": mad,
        "depth_filter_band_m": band,
        "depth_filter_retained_fraction": float(
            np.count_nonzero(support) / np.count_nonzero(valid)
        ),
    }


def _write_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No predicted-mask crop rows were produced.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_predicted_mask_crops(
    crop_labels_csv: str | Path,
    weights: str | Path,
    output_dir: str | Path,
    classes: dict[str, int],
    confidence: float = 0.25,
    minimum_iou: float = 0.1,
    image_size: int = 640,
    device: str | None = None,
    depth_mad_scale: float = 3.0,
    minimum_depth_band_m: float = 0.08,
    maximum_depth_band_m: float = 0.25,
) -> dict[str, Any]:
    from ultralytics import YOLO

    rows = _read_rows(crop_labels_csv)
    weights_path = Path(weights).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO segmentation weights do not exist: {weights_path}")

    output = Path(output_dir).expanduser().resolve()
    mask_dir = output / "target_masks"
    depth_support_mask_dir = output / "depth_support_masks"
    masked_depth_dir = output / "masked_depth"
    bbox_mask_dir = output / "bbox_masks"
    bbox_depth_support_mask_dir = output / "bbox_depth_support_masks"
    bbox_masked_depth_dir = output / "bbox_masked_depth"
    mask_dir.mkdir(parents=True, exist_ok=True)
    depth_support_mask_dir.mkdir(parents=True, exist_ok=True)
    masked_depth_dir.mkdir(parents=True, exist_ok=True)
    bbox_mask_dir.mkdir(parents=True, exist_ok=True)
    bbox_depth_support_mask_dir.mkdir(parents=True, exist_ok=True)
    bbox_masked_depth_dir.mkdir(parents=True, exist_ok=True)
    for directory in (
        mask_dir,
        depth_support_mask_dir,
        masked_depth_dir,
        bbox_mask_dir,
        bbox_depth_support_mask_dir,
        bbox_masked_depth_dir,
    ):
        for existing in directory.iterdir():
            if existing.is_file() or existing.is_symlink():
                existing.unlink()

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_image_path"]].append(row)

    source_paths = list(grouped)
    model = YOLO(str(weights_path))
    output_rows: list[dict[str, Any]] = []
    bbox_output_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    matched_ious: list[float] = []
    class_total: Counter[str] = Counter()
    class_matched: Counter[str] = Counter()

    for result_index, source_value in enumerate(source_paths, start=1):
        result = model.predict(
            source=source_value,
            conf=confidence,
            imgsz=image_size,
            device=device,
            verbose=False,
        )[0]
        image_rows = grouped[source_value]
        used_indices: set[int] = set()

        if result.boxes is None:
            predicted_boxes = np.empty((0, 4), dtype=np.float64)
            predicted_classes = np.empty((0,), dtype=np.int64)
            predicted_confidences = np.empty((0,), dtype=np.float64)
        else:
            predicted_boxes = result.boxes.xyxy.cpu().numpy()
            predicted_classes = result.boxes.cls.cpu().numpy().astype(np.int64)
            predicted_confidences = result.boxes.conf.cpu().numpy()
        polygons = result.masks.xy if result.masks is not None else []

        for row in image_rows:
            semantic_class = row["semantic_class"].strip()
            class_total[semantic_class] += 1
            if semantic_class not in classes:
                raise ValueError(f"Unknown semantic class: {semantic_class}")

            gt_bbox = np.asarray([
                float(row["bbox_x1"]),
                float(row["bbox_y1"]),
                float(row["bbox_x2"]),
                float(row["bbox_y2"]),
            ])
            prediction_index, match_iou = select_prediction_index(
                ground_truth_bbox=gt_bbox,
                ground_truth_class_id=classes[semantic_class],
                predicted_boxes=predicted_boxes,
                predicted_class_ids=predicted_classes,
                used_indices=used_indices,
            )

            failure_reason = None
            if prediction_index is None:
                failure_reason = "no_class_match"
            elif prediction_index >= len(polygons):
                failure_reason = "missing_predicted_mask"
            elif match_iou < minimum_iou:
                failure_reason = "bbox_iou_below_threshold"

            if failure_reason is not None:
                failures.append({
                    "frame_id": row["frame_id"],
                    "object_index": row["object_index"],
                    "semantic_class": semantic_class,
                    "reason": failure_reason,
                    "best_bbox_iou": match_iou,
                })
                continue

            assert prediction_index is not None
            used_indices.add(prediction_index)
            crop_box = (
                int(float(row["crop_left"])),
                int(float(row["crop_top"])),
                int(float(row["crop_right"])),
                int(float(row["crop_bottom"])),
            )
            try:
                predicted_mask = rasterize_polygon_crop(
                    polygon_xy=np.asarray(polygons[prediction_index]),
                    crop_box=crop_box,
                )
            except ValueError:
                failures.append({
                    "frame_id": row["frame_id"],
                    "object_index": row["object_index"],
                    "semantic_class": semantic_class,
                    "reason": "invalid_predicted_polygon",
                    "best_bbox_iou": match_iou,
                })
                continue

            depth_path = Path(row["depth_crop_path"]).expanduser().resolve()
            if not depth_path.exists():
                raise FileNotFoundError(f"Depth crop does not exist: {depth_path}")
            depth = np.load(depth_path, allow_pickle=False).astype(np.float32, copy=False)
            if predicted_mask.shape != depth.shape:
                predicted_mask = np.asarray(
                    Image.fromarray(predicted_mask.astype(np.uint8) * 255).resize(
                        (depth.shape[1], depth.shape[0]),
                        resample=Image.Resampling.NEAREST,
                    )
                ) > 0

            try:
                predicted_bbox_mask = rasterize_bbox_crop(
                    bbox_xyxy=predicted_boxes[prediction_index],
                    crop_box=crop_box,
                )
            except ValueError:
                predicted_bbox_mask = np.zeros_like(predicted_mask)
            if predicted_bbox_mask.shape != depth.shape:
                predicted_bbox_mask = np.asarray(
                    Image.fromarray(predicted_bbox_mask.astype(np.uint8) * 255).resize(
                        (depth.shape[1], depth.shape[0]),
                        resample=Image.Resampling.NEAREST,
                    )
                ) > 0

            try:
                depth_support_mask, depth_filter = filter_predicted_mask_depth(
                    depth=depth,
                    predicted_mask=predicted_mask,
                    mad_scale=depth_mad_scale,
                    minimum_band_m=minimum_depth_band_m,
                    maximum_band_m=maximum_depth_band_m,
                )
            except ValueError:
                failures.append({
                    "frame_id": row["frame_id"],
                    "object_index": row["object_index"],
                    "semantic_class": semantic_class,
                    "reason": "invalid_depth_support",
                    "best_bbox_iou": match_iou,
                })
                continue

            masked_depth = np.where(depth_support_mask, depth, 0.0).astype(np.float32)
            bbox_depth_support_mask, bbox_depth_filter = filter_predicted_mask_depth(
                depth=depth,
                predicted_mask=predicted_bbox_mask,
                mad_scale=depth_mad_scale,
                minimum_band_m=minimum_depth_band_m,
                maximum_band_m=maximum_depth_band_m,
            )
            bbox_masked_depth = np.where(
                bbox_depth_support_mask,
                depth,
                0.0,
            ).astype(np.float32)
            output_name = (
                f"{row['frame_id']}_obj{int(row['object_index']):02d}_"
                f"{semantic_class}.png"
            )
            mask_path = mask_dir / output_name
            depth_support_mask_path = depth_support_mask_dir / output_name
            masked_depth_path = masked_depth_dir / output_name.replace(".png", ".npy")
            bbox_mask_path = bbox_mask_dir / output_name
            bbox_depth_support_mask_path = bbox_depth_support_mask_dir / output_name
            bbox_masked_depth_path = bbox_masked_depth_dir / output_name.replace(
                ".png",
                ".npy",
            )
            Image.fromarray(predicted_mask.astype(np.uint8) * 255, mode="L").save(mask_path)
            Image.fromarray(
                depth_support_mask.astype(np.uint8) * 255,
                mode="L",
            ).save(depth_support_mask_path)
            np.save(masked_depth_path, masked_depth)
            Image.fromarray(
                predicted_bbox_mask.astype(np.uint8) * 255,
                mode="L",
            ).save(bbox_mask_path)
            Image.fromarray(
                bbox_depth_support_mask.astype(np.uint8) * 255,
                mode="L",
            ).save(bbox_depth_support_mask_path)
            np.save(bbox_masked_depth_path, bbox_masked_depth)

            updated: dict[str, Any] = dict(row)
            updated.update({
                "target_mask_path": str(mask_path),
                "depth_support_mask_path": str(depth_support_mask_path),
                "masked_depth_crop_path": str(masked_depth_path),
                "target_mask_fraction": float(np.mean(predicted_mask)),
                "depth_support_mask_fraction": float(np.mean(depth_support_mask)),
                "mask_source": "predicted_yolo_segmentation",
                "segmentation_confidence": float(predicted_confidences[prediction_index]),
                "segmentation_bbox_iou": match_iou,
                "segmentation_prediction_index": prediction_index,
            })
            updated.update(depth_filter)
            updated.update(
                _target_depth_statistics(masked_depth, depth_support_mask)
            )
            output_rows.append(updated)

            bbox_updated: dict[str, Any] = dict(row)
            bbox_updated.update({
                "target_mask_path": str(bbox_mask_path),
                "depth_support_mask_path": str(bbox_depth_support_mask_path),
                "masked_depth_crop_path": str(bbox_masked_depth_path),
                "target_mask_fraction": float(np.mean(predicted_bbox_mask)),
                "depth_support_mask_fraction": float(
                    np.mean(bbox_depth_support_mask)
                ),
                "mask_source": "predicted_yolo_bbox",
                "segmentation_confidence": float(
                    predicted_confidences[prediction_index]
                ),
                "segmentation_bbox_iou": match_iou,
                "segmentation_prediction_index": prediction_index,
            })
            bbox_updated.update(bbox_depth_filter)
            bbox_updated.update(
                _target_depth_statistics(
                    bbox_masked_depth,
                    bbox_depth_support_mask,
                )
            )
            bbox_output_rows.append(bbox_updated)
            matched_ious.append(match_iou)
            class_matched[semantic_class] += 1

        if device == "mps" and result_index % 50 == 0:
            import torch

            torch.mps.empty_cache()

    labels_output = output / "crop_labels.csv"
    _write_rows(output_rows, labels_output)
    bbox_labels_output = output / "bbox_crop_labels.csv"
    _write_rows(bbox_output_rows, bbox_labels_output)
    failure_output = output / "prediction_failures.json"
    failure_output.write_text(json.dumps(failures, indent=2) + "\n")

    per_class = {
        class_name: {
            "total": class_total[class_name],
            "matched": class_matched[class_name],
            "coverage": class_matched[class_name] / class_total[class_name],
        }
        for class_name in sorted(class_total)
    }
    report = {
        "model_family": "yolo_instance_segmentation",
        "weights": str(weights_path),
        "crop_labels_input": str(Path(crop_labels_csv).expanduser().resolve()),
        "crop_labels_output": str(labels_output),
        "bbox_crop_labels_output": str(bbox_labels_output),
        "num_input_rows": len(rows),
        "num_matched_rows": len(output_rows),
        "num_failed_rows": len(failures),
        "coverage": len(output_rows) / len(rows),
        "mean_matched_bbox_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "minimum_matched_bbox_iou": min(matched_ious) if matched_ious else 0.0,
        "confidence_threshold": confidence,
        "minimum_match_iou": minimum_iou,
        "image_size": image_size,
        "device": device or "auto",
        "depth_filter": {
            "method": "predicted-mask median plus bounded MAD band",
            "mad_scale": depth_mad_scale,
            "minimum_band_m": minimum_depth_band_m,
            "maximum_band_m": maximum_depth_band_m,
        },
        "per_class": per_class,
        "failure_counts": dict(Counter(item["reason"] for item in failures)),
    }
    report_path = output / "prediction_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
