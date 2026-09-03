from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path
    project_code: Path
    unity_output_root: Path
    unity_dataset_path: Path
    yolo_dataset_output: Path
    yolo_runs_output: Path
    predictions_output: Path
    model_weights: Path
    base_model: str
    run_name: str
    epochs: int
    image_size: int
    batch: int
    classes: dict[str, int]
    ignore_labels: set[str]


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def load_config(config_path: str | Path) -> ProjectConfig:
    config_path = _path(config_path)
    with config_path.open("r") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    required = [
        "project_root",
        "project_code",
        "unity_output_root",
        "yolo_dataset_output",
        "yolo_runs_output",
        "predictions_output",
        "model_weights",
        "classes",
    ]
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"Missing required config keys in {config_path}: {missing}")

    classes = {str(name): int(class_id) for name, class_id in raw["classes"].items()}
    class_ids = sorted(classes.values())
    expected_ids = list(range(len(class_ids)))
    if class_ids != expected_ids:
        raise ValueError(f"Class IDs must be zero-based and contiguous. Got: {class_ids}")

    unity_output_root = _path(raw["unity_output_root"])
    unity_dataset_path = _path(raw.get("unity_dataset_path", unity_output_root))

    return ProjectConfig(
        project_root=_path(raw["project_root"]),
        project_code=_path(raw["project_code"]),
        unity_output_root=unity_output_root,
        unity_dataset_path=unity_dataset_path,
        yolo_dataset_output=_path(raw["yolo_dataset_output"]),
        yolo_runs_output=_path(raw["yolo_runs_output"]),
        predictions_output=_path(raw["predictions_output"]),
        model_weights=_path(raw["model_weights"]),
        base_model=str(raw.get("base_model", "yolov8n.pt")),
        run_name=str(raw.get("run_name", "train1")),
        epochs=int(raw.get("epochs", 50)),
        image_size=int(raw.get("image_size", 640)),
        batch=int(raw.get("batch", 8)),
        classes=classes,
        ignore_labels={str(label) for label in raw.get("ignore_labels", [])},
    )
