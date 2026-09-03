from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import os
import shutil
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image

from .utils import ensure_split


@dataclass(frozen=True)
class SegmentationExportSummary:
    output_path: Path
    split_counts: dict[str, int]
    object_counts: dict[str, int]
    negative_frame_count: int
    skipped_objects: int
    fragmented_masks: int
    minimum_largest_component_fraction: float
    mean_largest_component_fraction: float


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
        "target_mask_path",
        "semantic_class",
        "frame_id",
        "crop_left",
        "crop_top",
        "source_image_width",
        "source_image_height",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Crop label CSV is missing required columns: {missing}")

    return rows


def _read_null_frame_rows(crop_labels_csv: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(crop_labels_csv).expanduser().resolve().with_name(
        "frame_manifest.csv"
    )
    if not manifest_path.exists():
        return []

    with manifest_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    required = {"frame_id", "source_image_path", "is_null_sample"}
    if rows:
        missing = sorted(required - set(rows[0]))
        if missing:
            raise ValueError(
                f"Frame manifest is missing required columns: {missing}"
            )

    return [
        row
        for row in rows
        if str(row["is_null_sample"]).strip().lower()
        in {"1", "true", "yes"}
    ]


def _largest_external_contour(mask: np.ndarray) -> tuple[np.ndarray, int, float]:
    binary = np.asarray(mask > 0, dtype=np.uint8)
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    contours = [contour for contour in contours if cv2.contourArea(contour) > 0.0]
    if not contours:
        raise ValueError("Target mask has no non-empty external contour.")

    component_areas = [int(np.count_nonzero(cv2.drawContours(
        np.zeros_like(binary),
        [contour],
        contourIdx=-1,
        color=1,
        thickness=cv2.FILLED,
    ))) for contour in contours]
    largest_index = int(np.argmax(component_areas))
    retained_fraction = component_areas[largest_index] / max(1, sum(component_areas))
    return contours[largest_index], len(contours), float(retained_fraction)


def _simplify_contour(contour: np.ndarray, tolerance: float) -> np.ndarray:
    if tolerance < 0.0:
        raise ValueError("Polygon simplification tolerance must be non-negative.")

    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = tolerance * perimeter
    simplified = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
    points = simplified.reshape(-1, 2).astype(np.float64)

    if len(points) < 3:
        points = contour.reshape(-1, 2).astype(np.float64)
    if len(points) < 3:
        raise ValueError("Target mask contour has fewer than three polygon points.")

    return points


def mask_to_yolo_polygon(
    mask: np.ndarray,
    crop_left: float,
    crop_top: float,
    image_width: float,
    image_height: float,
    simplification_tolerance: float = 0.001,
) -> tuple[list[float], int, float]:
    if image_width <= 0.0 or image_height <= 0.0:
        raise ValueError("Source image dimensions must be positive.")

    contour, component_count, retained_fraction = _largest_external_contour(mask)
    points = _simplify_contour(contour, simplification_tolerance)
    points[:, 0] = (points[:, 0] + crop_left) / image_width
    points[:, 1] = (points[:, 1] + crop_top) / image_height
    points = np.clip(points, 0.0, 1.0)

    return points.reshape(-1).tolist(), component_count, retained_fraction


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_segmentation_data_yaml(
    output_path: str | Path,
    classes: dict[str, int],
) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    names_by_id = {class_id: name for name, class_id in classes.items()}
    expected_ids = list(range(len(names_by_id)))
    if sorted(names_by_id) != expected_ids:
        raise ValueError(
            f"Class IDs must be zero-based and contiguous. Got: {sorted(names_by_id)}"
        )

    lines = [
        f"path: {output}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for class_id in expected_ids:
        lines.append(f"  {class_id}: {names_by_id[class_id]}")

    yaml_path = output / "data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


def export_yolo_segmentation_split(
    crop_labels_csv: str | Path,
    output_path: str | Path,
    split: str,
    classes: dict[str, int],
    simplification_tolerance: float = 0.001,
    minimum_component_fraction: float = 0.75,
    clear_split: bool = False,
) -> SegmentationExportSummary:
    split = ensure_split(split)
    output = Path(output_path).expanduser().resolve()
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    if clear_split:
        for directory in (image_dir, label_dir):
            for path in directory.iterdir():
                if path.is_file() or path.is_symlink():
                    path.unlink()

    rows = _read_rows(crop_labels_csv)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_image_path"]].append(row)

    image_count = 0
    object_count = 0
    skipped_objects = 0
    fragmented_masks = 0
    retained_fractions: list[float] = []
    exported_source_images: set[Path] = set()

    for source_value, image_rows in grouped.items():
        source_image = Path(source_value).expanduser().resolve()
        if not source_image.exists():
            raise FileNotFoundError(f"Source RGB image does not exist: {source_image}")

        frame_id = image_rows[0]["frame_id"]
        image_stem = f"{frame_id}_{source_image.stem}"
        image_destination = image_dir / f"{image_stem}{source_image.suffix.lower()}"
        label_destination = label_dir / f"{image_stem}.txt"
        label_lines: list[str] = []

        for row in image_rows:
            semantic_class = row["semantic_class"].strip()
            if semantic_class not in classes:
                raise ValueError(
                    f"Unknown semantic class '{semantic_class}' in {crop_labels_csv}."
                )

            mask_path = Path(row["target_mask_path"]).expanduser().resolve()
            if not mask_path.exists():
                raise FileNotFoundError(f"Target mask does not exist: {mask_path}")

            with Image.open(mask_path) as image:
                mask = np.asarray(image.convert("L")) > 0

            try:
                polygon, component_count, retained_fraction = mask_to_yolo_polygon(
                    mask=mask,
                    crop_left=float(row["crop_left"]),
                    crop_top=float(row["crop_top"]),
                    image_width=float(row["source_image_width"]),
                    image_height=float(row["source_image_height"]),
                    simplification_tolerance=simplification_tolerance,
                )
            except ValueError:
                skipped_objects += 1
                continue

            if retained_fraction < minimum_component_fraction:
                skipped_objects += 1
                continue

            if component_count > 1:
                fragmented_masks += 1
            retained_fractions.append(retained_fraction)

            coordinates = " ".join(f"{value:.6f}" for value in polygon)
            label_lines.append(f"{classes[semantic_class]} {coordinates}")

        if not label_lines:
            continue

        _link_or_copy(source_image, image_destination)
        label_destination.write_text("\n".join(label_lines) + "\n")
        exported_source_images.add(source_image)
        image_count += 1
        object_count += len(label_lines)

    negative_frame_count = 0
    for null_row in _read_null_frame_rows(crop_labels_csv):
        source_image = Path(null_row["source_image_path"]).expanduser().resolve()
        if not source_image.exists():
            raise FileNotFoundError(
                f"Null-frame RGB image does not exist: {source_image}"
            )
        if source_image in exported_source_images:
            raise ValueError(
                f"Frame is marked null but also has target objects: {source_image}"
            )

        frame_id = null_row["frame_id"]
        image_stem = f"{frame_id}_{source_image.stem}"
        image_destination = image_dir / f"{image_stem}{source_image.suffix.lower()}"
        label_destination = label_dir / f"{image_stem}.txt"

        _link_or_copy(source_image, image_destination)
        label_destination.write_text("")
        exported_source_images.add(source_image)
        image_count += 1
        negative_frame_count += 1

    write_segmentation_data_yaml(output, classes)

    summary = SegmentationExportSummary(
        output_path=output,
        split_counts={split: image_count},
        object_counts={split: object_count},
        negative_frame_count=negative_frame_count,
        skipped_objects=skipped_objects,
        fragmented_masks=fragmented_masks,
        minimum_largest_component_fraction=(
            min(retained_fractions) if retained_fractions else 0.0
        ),
        mean_largest_component_fraction=(
            float(np.mean(retained_fractions)) if retained_fractions else 0.0
        ),
    )
    report_path = output / f"export_summary_{split}.json"
    payload: dict[str, Any] = asdict(summary)
    payload["output_path"] = str(summary.output_path)
    report_path.write_text(json.dumps(payload, indent=2) + "\n")
    return summary


def export_yolo_segmentation_dataset(
    crop_label_splits: Iterable[tuple[str, str | Path]],
    output_path: str | Path,
    classes: dict[str, int],
    simplification_tolerance: float = 0.001,
    minimum_component_fraction: float = 0.75,
    clear_splits: bool = False,
) -> dict[str, SegmentationExportSummary]:
    summaries: dict[str, SegmentationExportSummary] = {}
    for split, crop_labels_csv in crop_label_splits:
        normalized_split = ensure_split(split)
        if normalized_split in summaries:
            raise ValueError(f"Duplicate segmentation split: {normalized_split}")
        summaries[normalized_split] = export_yolo_segmentation_split(
            crop_labels_csv=crop_labels_csv,
            output_path=output_path,
            split=normalized_split,
            classes=classes,
            simplification_tolerance=simplification_tolerance,
            minimum_component_fraction=minimum_component_fraction,
            clear_split=clear_splits,
        )
    return summaries
