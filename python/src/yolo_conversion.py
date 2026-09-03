from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil

from .config import ProjectConfig
from .unity_dataset import UnityFrame, dataset_path_from_name, iter_unity_frames
from .utils import clear_directory_contents, ensure_split


@dataclass(frozen=True)
class ConversionSummary:
    dataset_path: Path
    output_path: Path
    split_counts: dict[str, int]
    skipped_empty: int


def _frame_output_stem(frame: UnityFrame) -> str:
    return f"{frame.dataset_name}_{frame.sequence_name}_{frame.step_name}"


def _to_yolo_line(label: str, origin: tuple[float, float], dimension: tuple[float, float], width: float, height: float, classes: dict[str, int]) -> str:
    x, y = origin
    w, h = dimension
    x_center = (x + w / 2) / width
    y_center = (y + h / 2) / height
    w_norm = w / width
    h_norm = h / height

    values = [x_center, y_center, w_norm, h_norm]
    values = [max(0.0, min(1.0, float(value))) for value in values]
    return f"{classes[label]} " + " ".join(f"{value:.6f}" for value in values)


def write_data_yaml(config: ProjectConfig) -> Path:
    output_path = config.yolo_dataset_output
    output_path.mkdir(parents=True, exist_ok=True)
    names_by_id = {class_id: name for name, class_id in config.classes.items()}

    lines = [
        f"path: {output_path}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for class_id in sorted(names_by_id):
        lines.append(f"  {class_id}: {names_by_id[class_id]}")

    data_yaml = output_path / "data.yaml"
    data_yaml.write_text("\n".join(lines) + "\n")
    return data_yaml


def _split_frames(frames: list[UnityFrame], split: str | None, ratios: tuple[float, float, float] | None, seed: int) -> dict[str, list[UnityFrame]]:
    if ratios is None:
        assert split is not None
        return {ensure_split(split): frames}

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0. Got: {ratios}")

    shuffled = list(frames)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def convert_dataset(
    config: ProjectConfig,
    dataset: str | Path | None = None,
    split: str | None = "train",
    ratios: tuple[float, float, float] | None = None,
    seed: int = 42,
    clear_split: bool = False,
) -> ConversionSummary:
    dataset_path = config.unity_dataset_path if dataset is None else dataset_path_from_name(config.unity_output_root, dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Unity dataset not found: {dataset_path}")

    frames = iter_unity_frames(dataset_path)
    grouped = _split_frames(frames, split, ratios, seed)
    split_counts = {name: 0 for name in grouped}
    skipped_empty = 0

    for split_name in grouped:
        image_dir = config.yolo_dataset_output / "images" / split_name
        label_dir = config.yolo_dataset_output / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        if clear_split:
            clear_directory_contents(image_dir)
            clear_directory_contents(label_dir)

    for split_name, split_frames in grouped.items():
        image_dir = config.yolo_dataset_output / "images" / split_name
        label_dir = config.yolo_dataset_output / "labels" / split_name

        for frame in split_frames:
            yolo_lines: list[str] = []
            for obj in frame.objects:
                if obj.label in config.ignore_labels:
                    continue
                if obj.label not in config.classes:
                    raise ValueError(
                        f"Unknown label '{obj.label}' in {frame.frame_json}. Add it to config/classes or config/ignore_labels."
                    )
                yolo_lines.append(_to_yolo_line(obj.label, obj.origin, obj.dimension, frame.width, frame.height, config.classes))

            if not yolo_lines:
                skipped_empty += 1
                continue

            stem = _frame_output_stem(frame)
            image_name = f"{stem}{frame.image_path.suffix.lower()}"
            label_name = f"{stem}.txt"
            shutil.copy2(frame.image_path, image_dir / image_name)
            (label_dir / label_name).write_text("\n".join(yolo_lines) + "\n")
            split_counts[split_name] += 1

    write_data_yaml(config)
    return ConversionSummary(dataset_path, config.yolo_dataset_output, split_counts, skipped_empty)
