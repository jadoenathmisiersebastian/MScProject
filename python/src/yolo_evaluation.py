from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import ProjectConfig
from .utils import IMAGE_EXTENSIONS, ensure_split


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def evaluate_yolo(
    config: ProjectConfig,
    weights: str | Path | None = None,
    split: str = "test",
    run_name: str | None = None,
    image_size: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    data_yaml: str | Path | None = None,
    negative_confidence: float = 0.25,
    exist_ok: bool = False,
) -> Path:
    from ultralytics import YOLO, __version__ as ultralytics_version

    split = ensure_split(split)
    weights_path = Path(weights).expanduser().resolve() if weights else config.model_weights
    resolved_data_yaml = (
        Path(data_yaml).expanduser().resolve()
        if data_yaml is not None
        else config.yolo_dataset_output / "data.yaml"
    )
    dataset_root = resolved_data_yaml.parent
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    if not resolved_data_yaml.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found: {resolved_data_yaml}")
    if not image_dir.exists():
        raise FileNotFoundError(f"YOLO {split} image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"YOLO {split} label directory not found: {label_dir}")

    resolved_image_size = image_size if image_size is not None else config.image_size
    resolved_batch = batch if batch is not None else config.batch
    resolved_run_name = run_name or f"yolo_{split}_evaluation"

    model = YOLO(str(weights_path))
    input_channels = int(model.model.yaml.get("channels", 3))
    metrics = model.val(
        data=str(resolved_data_yaml),
        split=split,
        imgsz=resolved_image_size,
        batch=resolved_batch,
        project=str(config.yolo_runs_output),
        name=resolved_run_name,
        device=device,
        plots=True,
        exist_ok=exist_ok,
    )

    result_values = metrics.results_dict
    per_class = [
        {key: _to_builtin(value) for key, value in row.items()}
        for row in metrics.summary()
    ]
    output_dir = Path(metrics.save_dir).expanduser().resolve()

    empty_label_files = [
        path
        for path in sorted(label_dir.glob("*.txt"))
        if not path.read_text().strip()
    ]
    negative_false_positives: list[dict[str, Any]] = []
    total_negative_predictions = 0

    for label_path in empty_label_files:
        image_path = next(
            (
                image_dir / f"{label_path.stem}{extension}"
                for extension in sorted(IMAGE_EXTENSIONS)
                if (image_dir / f"{label_path.stem}{extension}").exists()
            ),
            None,
        )
        if image_path is None:
            continue

        prediction_source: str | np.ndarray = str(image_path)
        if input_channels > 3:
            sidecar_path = image_path.with_suffix(".npy")
            if not sidecar_path.exists():
                raise FileNotFoundError(
                    f"{input_channels}-channel sidecar does not exist: {sidecar_path}"
                )
            prediction_source = np.load(sidecar_path, allow_pickle=False)

        prediction = model.predict(
            source=prediction_source,
            imgsz=resolved_image_size,
            conf=negative_confidence,
            device=device,
            verbose=False,
        )[0]
        prediction_count = len(prediction.boxes) if prediction.boxes is not None else 0
        total_negative_predictions += prediction_count

        if prediction_count > 0:
            confidences = prediction.boxes.conf
            maximum_confidence = (
                float(confidences.max().item())
                if confidences is not None and len(confidences) > 0
                else 0.0
            )
            negative_false_positives.append({
                "image": str(image_path),
                "num_predictions": prediction_count,
                "maximum_confidence": maximum_confidence,
            })

    image_count = sum(
        path.suffix.lower() in IMAGE_EXTENSIONS
        for path in image_dir.iterdir()
        if path.is_file()
    )
    label_count = sum(path.suffix.lower() == ".txt" for path in label_dir.iterdir() if path.is_file())

    box_metrics = {
        "precision": float(result_values["metrics/precision(B)"]),
        "recall": float(result_values["metrics/recall(B)"]),
        "map50": float(result_values["metrics/mAP50(B)"]),
        "map50_95": float(result_values["metrics/mAP50-95(B)"]),
        "fitness": float(result_values["fitness"]),
    }
    mask_metrics = None
    if "metrics/precision(M)" in result_values:
        mask_metrics = {
            "precision": float(result_values["metrics/precision(M)"]),
            "recall": float(result_values["metrics/recall(M)"]),
            "map50": float(result_values["metrics/mAP50(M)"]),
            "map50_95": float(result_values["metrics/mAP50-95(M)"]),
        }

    report = {
        "evaluation_split": split,
        "weights": str(weights_path),
        "data_yaml": str(resolved_data_yaml),
        "image_dir": str(image_dir.resolve()),
        "label_dir": str(label_dir.resolve()),
        "num_images": image_count,
        "num_label_files": label_count,
        "image_size": resolved_image_size,
        "input_channels": input_channels,
        "batch_size": resolved_batch,
        "device": device or "auto",
        "ultralytics_version": ultralytics_version,
        "overall_metrics": box_metrics,
        "segmentation_metrics": mask_metrics,
        "negative_frame_metrics": {
            "confidence_threshold": negative_confidence,
            "num_negative_images": len(empty_label_files),
            "num_negative_images_with_predictions": len(negative_false_positives),
            "false_positive_frame_rate": (
                len(negative_false_positives) / len(empty_label_files)
                if empty_label_files
                else 0.0
            ),
            "num_false_positive_instances": total_negative_predictions,
            "false_positive_images": negative_false_positives,
        },
        "per_class_metrics": per_class,
        "speed_ms_per_image": {
            key: float(value)
            for key, value in metrics.speed.items()
        },
        "output_dir": str(output_dir),
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(report, f, indent=2)

    return metrics_path
