from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def binary_mask_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if truth.shape != prediction.shape:
        raise ValueError(
            f"Mask dimensions do not match: {truth.shape} and {prediction.shape}"
        )

    true_positive = int(np.count_nonzero(truth & prediction))
    false_positive = int(np.count_nonzero(~truth & prediction))
    false_negative = int(np.count_nonzero(truth & ~prediction))
    union = true_positive + false_positive + false_negative
    truth_count = true_positive + false_negative
    prediction_count = true_positive + false_positive

    return {
        "iou": true_positive / union if union else 1.0,
        "dice": (
            2.0 * true_positive / (truth_count + prediction_count)
            if truth_count + prediction_count
            else 1.0
        ),
        "precision": true_positive / prediction_count if prediction_count else 1.0,
        "recall": true_positive / truth_count if truth_count else 1.0,
    }


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Crop label CSV does not exist: {resolved}")
    with resolved.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Crop label CSV contains no rows: {resolved}")
    return rows


def _mask(path: str) -> np.ndarray:
    with Image.open(Path(path).expanduser().resolve()) as image:
        return np.asarray(image.convert("L")) > 0


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "num_rows": len(rows),
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "median_iou": float(np.median([row["iou"] for row in rows])),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "mean_precision": float(np.mean([row["precision"] for row in rows])),
        "mean_recall": float(np.mean([row["recall"] for row in rows])),
    }


def _error_overlay(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    overlay = np.zeros((*truth.shape, 3), dtype=np.uint8)
    overlay[truth & prediction] = np.asarray([40, 170, 70], dtype=np.uint8)
    overlay[~truth & prediction] = np.asarray([220, 65, 55], dtype=np.uint8)
    overlay[truth & ~prediction] = np.asarray([60, 110, 210], dtype=np.uint8)
    return overlay


def generate_segmentation_figures(
    ground_truth_crop_labels: str | Path,
    predicted_crop_labels: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    truth_rows = _read_rows(ground_truth_crop_labels)
    predicted_rows = _read_rows(predicted_crop_labels)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    truth_by_key = {
        (row["frame_id"], row["object_index"]): row
        for row in truth_rows
    }
    metrics: list[dict[str, Any]] = []
    for predicted in predicted_rows:
        key = (predicted["frame_id"], predicted["object_index"])
        truth = truth_by_key.get(key)
        if truth is None:
            continue
        truth_mask = _mask(truth["target_mask_path"])
        predicted_mask = _mask(predicted["target_mask_path"])
        row: dict[str, Any] = {
            "frame_id": key[0],
            "object_index": key[1],
            "semantic_class": predicted["semantic_class"],
            "crop_image_path": truth["crop_image_path"],
            "truth_mask_path": truth["target_mask_path"],
            "predicted_mask_path": predicted["target_mask_path"],
        }
        row.update(binary_mask_metrics(truth_mask, predicted_mask))
        metrics.append(row)

    if not metrics:
        raise ValueError("No matching truth and predicted mask rows were found.")

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        by_class[row["semantic_class"]].append(row)

    summary = {
        "ground_truth_crop_labels": str(
            Path(ground_truth_crop_labels).expanduser().resolve()
        ),
        "predicted_crop_labels": str(
            Path(predicted_crop_labels).expanduser().resolve()
        ),
        "num_ground_truth_rows": len(truth_rows),
        "num_predicted_rows": len(predicted_rows),
        "coverage": len(metrics) / len(truth_rows),
        "overall": _aggregate(metrics),
        "per_class": {
            class_name: _aggregate(class_rows)
            for class_name, class_rows in sorted(by_class.items())
        },
    }
    summary_path = output / "mask_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    ordered = sorted(metrics, key=lambda row: row["iou"])
    selected = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    row_names = ("Worst", "Median", "Best")
    figure, axes = plt.subplots(3, 4, figsize=(12, 8.5))
    for row_index, (row_name, row) in enumerate(zip(row_names, selected, strict=True)):
        with Image.open(row["crop_image_path"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        truth_mask = _mask(row["truth_mask_path"])
        predicted_mask = _mask(row["predicted_mask_path"])
        images = (
            rgb,
            truth_mask,
            predicted_mask,
            _error_overlay(truth_mask, predicted_mask),
        )
        for column, visual in enumerate(images):
            axes[row_index, column].imshow(
                visual,
                cmap="gray" if visual.ndim == 2 else None,
            )
            axes[row_index, column].axis("off")
        axes[row_index, 0].text(
            0.02,
            0.05,
            f"{row_name} | {row['semantic_class']}\nIoU {row['iou']:.3f}",
            transform=axes[row_index, 0].transAxes,
            color="white",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none"},
        )
    for axis, title in zip(
        axes[0],
        ("RGB Crop", "Unity Mask", "Predicted Mask", "TP / FP / FN"),
        strict=True,
    ):
        axis.set_title(title)
    figure.suptitle("Predicted Instance-Mask Examples")
    figure.tight_layout(rect=(0.04, 0.02, 1.0, 0.96))
    examples_path = output / "predicted_mask_examples.png"
    figure.savefig(examples_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.hist([row["iou"] for row in metrics], bins=20, color="#4e79a7", alpha=0.85)
    axis.axvline(summary["overall"]["mean_iou"], color="#e15759", linestyle="--", label="Mean")
    axis.set_xlabel("Mask IoU")
    axis.set_ylabel("Number Of Objects")
    axis.set_title("Predicted Mask IoU Distribution")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    distribution_path = output / "mask_iou_distribution.png"
    figure.savefig(distribution_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return [summary_path, examples_path, distribution_path]
