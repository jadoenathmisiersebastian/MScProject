from __future__ import annotations

from pathlib import Path

from .config import ProjectConfig


def train_yolo(config: ProjectConfig, model: str | None = None, run_name: str | None = None, epochs: int | None = None, image_size: int | None = None, batch: int | None = None, device: str | None = None, data_yaml: str | Path | None = None, exist_ok: bool = False):
    from ultralytics import YOLO

    resolved_data_yaml = (
        Path(data_yaml).expanduser().resolve()
        if data_yaml is not None
        else config.yolo_dataset_output / "data.yaml"
    )
    if not resolved_data_yaml.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found: {resolved_data_yaml}")

    yolo_model = YOLO(model or config.base_model)
    return yolo_model.train(
        data=str(resolved_data_yaml),
        epochs=epochs or config.epochs,
        imgsz=image_size or config.image_size,
        batch=batch or config.batch,
        project=str(config.yolo_runs_output),
        name=run_name or config.run_name,
        device=device,
        exist_ok=exist_ok,
    )
