from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import os
import shutil
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from .utils import ensure_split


DEPTH_ANNOTATION_TYPE = "type.unity.com/unity.solo.DepthAnnotation"


@dataclass(frozen=True)
class RGBDSegmentationExportSummary:
    output_path: Path
    split: str
    num_images: int
    num_negative_images: int
    num_written: int
    num_reused: int
    output_long_side: int
    depth_min_m: float
    depth_max_m: float
    mean_valid_depth_fraction: float


def encode_metric_depth(
    depth_m: np.ndarray,
    depth_min_m: float = 0.3,
    depth_max_m: float = 3.0,
) -> np.ndarray:
    """Encode metric depth as uint8, reserving zero for invalid pixels."""
    if depth_max_m <= depth_min_m:
        raise ValueError("depth_max_m must be greater than depth_min_m.")

    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    encoded = np.zeros(depth.shape, dtype=np.uint8)
    if not np.any(valid):
        return encoded

    normalized = np.clip(
        (depth[valid] - depth_min_m) / (depth_max_m - depth_min_m),
        0.0,
        1.0,
    )
    encoded[valid] = np.rint(normalized * 254.0).astype(np.uint8) + 1
    return encoded


def decode_metric_depth(
    encoded: np.ndarray,
    depth_min_m: float = 0.3,
    depth_max_m: float = 3.0,
) -> np.ndarray:
    """Decode a fourth-channel depth image for diagnostics and tests."""
    if depth_max_m <= depth_min_m:
        raise ValueError("depth_max_m must be greater than depth_min_m.")

    values = np.asarray(encoded, dtype=np.uint8)
    valid = values > 0
    depth = np.full(values.shape, np.nan, dtype=np.float32)
    depth[valid] = depth_min_m + (
        (values[valid].astype(np.float32) - 1.0) / 254.0
    ) * (depth_max_m - depth_min_m)
    return depth


def read_metric_depth(path: str | Path) -> np.ndarray:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Metric depth file does not exist: {resolved}")
    if resolved.suffix.lower() == ".npy":
        return np.asarray(np.load(resolved, allow_pickle=False), dtype=np.float32)

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    encoded = cv2.imread(str(resolved), cv2.IMREAD_UNCHANGED)
    if encoded is None:
        raise ValueError(f"Could not read metric depth: {resolved}")
    if encoded.ndim == 2:
        depth = encoded
    elif encoded.ndim == 3 and encoded.shape[2] >= 3:
        # OpenCV reads BGR(A); Unity writes metric depth into the R channel.
        depth = encoded[:, :, 2]
    else:
        raise ValueError(f"Unsupported depth shape {encoded.shape}: {resolved}")
    return np.asarray(depth, dtype=np.float32)


def _resized_shape(width: int, height: int, long_side: int) -> tuple[int, int]:
    if width <= 0 or height <= 0 or long_side <= 0:
        raise ValueError("Image dimensions and long_side must be positive.")
    scale = long_side / max(width, height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _resize_depth(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    weighted = np.where(valid, depth, 0.0).astype(np.float32)
    resized_weighted = cv2.resize(weighted, size, interpolation=cv2.INTER_LINEAR)
    resized_validity = cv2.resize(
        valid.astype(np.float32), size, interpolation=cv2.INTER_LINEAR
    )
    resized = np.full(resized_weighted.shape, np.nan, dtype=np.float32)
    retained = resized_validity >= 0.5
    resized[retained] = resized_weighted[retained] / np.maximum(
        resized_validity[retained], 1e-6
    )
    return resized


def build_rgbd_array(
    rgb_path: str | Path,
    depth_path: str | Path,
    long_side: int = 640,
    depth_min_m: float = 0.3,
    depth_max_m: float = 3.0,
) -> tuple[np.ndarray, float]:
    """Build an HWC RGB-D uint8 tensor accepted by Ultralytics multispectral input."""
    rgb_path = Path(rgb_path).expanduser().resolve()
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image does not exist: {rgb_path}")

    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    target_size = _resized_shape(rgb.shape[1], rgb.shape[0], long_side)
    resized_rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)

    depth = read_metric_depth(depth_path)
    resized_depth = _resize_depth(depth, target_size)
    encoded_depth = encode_metric_depth(
        resized_depth,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    rgbd = np.concatenate([resized_rgb, encoded_depth[..., None]], axis=2)
    valid_fraction = float(np.mean(encoded_depth > 0))
    return np.ascontiguousarray(rgbd, dtype=np.uint8), valid_fraction


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Frame manifest does not exist: {resolved}")
    with resolved.open(newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"frame_id", "source_image_path", "is_null_sample"}
    if not rows:
        raise ValueError(f"Frame manifest is empty: {resolved}")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Frame manifest is missing columns: {missing}")
    return rows


def _depth_path_for_image(source_image: Path) -> Path:
    name = source_image.name
    if not name.endswith(".camera.png"):
        raise ValueError(f"Expected a .camera.png source image, got: {source_image}")
    frame_data = source_image.with_name(
        name.removesuffix(".camera.png") + ".frame_data.json"
    )
    if not frame_data.exists():
        raise FileNotFoundError(f"Frame data does not exist: {frame_data}")
    payload = json.loads(frame_data.read_text())
    captures = payload.get("captures", [])
    matching = [
        capture
        for capture in captures
        if capture.get("filename") == source_image.name
    ]
    if not matching and len(captures) == 1:
        matching = captures
    if len(matching) != 1:
        raise ValueError(f"Could not identify camera capture in {frame_data}")

    annotations = matching[0].get("annotations", [])
    depth_annotations = [
        item for item in annotations if item.get("@type") == DEPTH_ANNOTATION_TYPE
    ]
    if len(depth_annotations) != 1 or not depth_annotations[0].get("filename"):
        raise ValueError(f"Expected one depth annotation in {frame_data}")
    return frame_data.parent / depth_annotations[0]["filename"]


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, array, allow_pickle=False)
    os.replace(temporary, path)


def _write_data_yaml(
    output_path: Path,
    source_data_yaml: Path,
) -> Path:
    import yaml

    source = yaml.safe_load(source_data_yaml.read_text())
    names = source.get("names")
    if names is None:
        raise ValueError(f"Source data YAML has no names mapping: {source_data_yaml}")
    payload = {
        "path": str(output_path),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "channels": 4,
        "names": names,
    }
    yaml_path = output_path / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return yaml_path


def export_rgbd_segmentation_split(
    source_dataset: str | Path,
    frame_manifest: str | Path,
    output_path: str | Path,
    split: str,
    long_side: int = 640,
    depth_min_m: float = 0.3,
    depth_max_m: float = 3.0,
    clear_split: bool = False,
    limit: int | None = None,
) -> RGBDSegmentationExportSummary:
    split = ensure_split(split)
    source_dataset = Path(source_dataset).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    source_images = source_dataset / "images" / split
    source_labels = source_dataset / "labels" / split
    if not source_images.exists() or not source_labels.exists():
        raise FileNotFoundError(f"Source YOLO split does not exist: {source_dataset} ({split})")

    image_output = output / "images" / split
    label_output = output / "labels" / split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    if clear_split:
        for directory in (image_output, label_output):
            for item in directory.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()

    mapping: dict[str, tuple[Path, Path]] = {}
    for row in _read_manifest(frame_manifest):
        source_image = Path(row["source_image_path"]).expanduser().resolve()
        exported_name = f"{row['frame_id']}_{source_image.stem}{source_image.suffix.lower()}"
        mapping[exported_name] = (source_image, _depth_path_for_image(source_image))

    image_paths = sorted(
        path for path in source_images.iterdir() if path.suffix.lower() == ".png"
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive when supplied.")
        image_paths = image_paths[:limit]

    written = 0
    reused = 0
    negatives = 0
    valid_fractions: list[float] = []
    for index, source_yolo_image in enumerate(image_paths, start=1):
        source_yolo_label = source_labels / f"{source_yolo_image.stem}.txt"
        if not source_yolo_label.exists():
            raise FileNotFoundError(f"YOLO label does not exist: {source_yolo_label}")
        if source_yolo_image.name not in mapping:
            raise KeyError(
                f"No frame-manifest source mapping for {source_yolo_image.name}"
            )
        original_rgb, original_depth = mapping[source_yolo_image.name]
        destination_image = image_output / source_yolo_image.name
        destination_label = label_output / source_yolo_label.name
        destination_npy = destination_image.with_suffix(".npy")

        _link_or_copy(source_yolo_image, destination_image)
        _link_or_copy(source_yolo_label, destination_label)
        negatives += int(not source_yolo_label.read_text().strip())

        if destination_npy.exists() and not clear_split:
            try:
                existing = np.load(destination_npy, mmap_mode="r", allow_pickle=False)
                if existing.ndim == 3 and existing.shape[2] == 4:
                    valid_fractions.append(float(np.mean(existing[:, :, 3] > 0)))
                    reused += 1
                    continue
            except (OSError, ValueError):
                pass

        rgbd, valid_fraction = build_rgbd_array(
            rgb_path=original_rgb,
            depth_path=original_depth,
            long_side=long_side,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
        )
        _atomic_save_npy(destination_npy, rgbd)
        valid_fractions.append(valid_fraction)
        written += 1
        if index % 250 == 0 or index == len(image_paths):
            print(
                f"RGB-D {split}: {index}/{len(image_paths)} "
                f"(written={written}, reused={reused})"
            )

    summary = RGBDSegmentationExportSummary(
        output_path=output,
        split=split,
        num_images=len(image_paths),
        num_negative_images=negatives,
        num_written=written,
        num_reused=reused,
        output_long_side=long_side,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        mean_valid_depth_fraction=float(np.mean(valid_fractions)) if valid_fractions else 0.0,
    )
    payload = asdict(summary)
    payload["output_path"] = str(output)
    (output / f"rgbd_export_summary_{split}.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    return summary


def export_rgbd_segmentation_dataset(
    source_dataset: str | Path,
    manifest_splits: Iterable[tuple[str, str | Path]],
    output_path: str | Path,
    long_side: int = 640,
    depth_min_m: float = 0.3,
    depth_max_m: float = 3.0,
    clear_splits: bool = False,
    limit_per_split: int | None = None,
) -> dict[str, RGBDSegmentationExportSummary]:
    source_dataset = Path(source_dataset).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, RGBDSegmentationExportSummary] = {}
    for split, manifest in manifest_splits:
        normalized = ensure_split(split)
        if normalized in summaries:
            raise ValueError(f"Duplicate RGB-D split: {normalized}")
        summaries[normalized] = export_rgbd_segmentation_split(
            source_dataset=source_dataset,
            frame_manifest=manifest,
            output_path=output,
            split=normalized,
            long_side=long_side,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
            clear_split=clear_splits,
            limit=limit_per_split,
        )

    _write_data_yaml(output, source_dataset / "data.yaml")
    encoding = {
        "channels": ["red", "green", "blue", "metric_depth"],
        "storage_dtype": "uint8",
        "invalid_depth_value": 0,
        "valid_depth_code_range": [1, 255],
        "depth_min_m": depth_min_m,
        "depth_max_m": depth_max_m,
        "depth_mapping": "linear_clipped",
        "output_long_side": long_side,
    }
    (output / "rgbd_encoding.json").write_text(json.dumps(encoding, indent=2) + "\n")
    return summaries
