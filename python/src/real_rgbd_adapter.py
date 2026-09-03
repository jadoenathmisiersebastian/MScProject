from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_binary_array(
    path: Path,
    dtype: np.dtype,
    width: int,
    height: int,
) -> np.ndarray:
    expected_values = width * height
    array = np.fromfile(path, dtype=dtype)

    if array.size != expected_values:
        raise ValueError(
            f"{path} contains {array.size} values; "
            f"expected {expected_values} for {width}x{height}."
        )

    return array.reshape(height, width)


def rotate_array(array: np.ndarray, rotation_degrees: int) -> np.ndarray:
    if rotation_degrees == 0:
        return array.copy()
    if rotation_degrees == 90:
        return np.rot90(array, k=3)
    if rotation_degrees == 180:
        return np.rot90(array, k=2)
    if rotation_degrees == 270:
        return np.rot90(array, k=1)

    raise ValueError("Rotation must be one of: 0, 90, 180, 270.")


def rotate_image(image: Image.Image, rotation_degrees: int) -> Image.Image:
    if rotation_degrees == 0:
        return image.copy()
    if rotation_degrees == 90:
        return image.transpose(Image.Transpose.ROTATE_270)
    if rotation_degrees == 180:
        return image.transpose(Image.Transpose.ROTATE_180)
    if rotation_degrees == 270:
        return image.transpose(Image.Transpose.ROTATE_90)

    raise ValueError("Rotation must be one of: 0, 90, 180, 270.")


def rotate_intrinsics(
    intrinsics: np.ndarray,
    width: int,
    height: int,
    rotation_degrees: int,
) -> tuple[np.ndarray, int, int]:
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    if rotation_degrees == 0:
        corrected = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ])
        return corrected, width, height

    if rotation_degrees == 180:
        corrected = np.array([
            [fx, 0.0, (width - 1) - cx],
            [0.0, fy, (height - 1) - cy],
            [0.0, 0.0, 1.0],
        ])
        return corrected, width, height

    if rotation_degrees == 90:
        corrected = np.array([
            [fy, 0.0, (height - 1) - cy],
            [0.0, fx, cx],
            [0.0, 0.0, 1.0],
        ])
        return corrected, height, width

    corrected = np.array([
        [fy, 0.0, cy],
        [0.0, fx, (width - 1) - cx],
        [0.0, 0.0, 1.0],
    ])
    return corrected, height, width


def scale_intrinsics(
    intrinsics: np.ndarray,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    scale_x = target_width / source_width
    scale_y = target_height / source_height

    return np.array([
        [
            intrinsics[0, 0] * scale_x,
            0.0,
            (intrinsics[0, 2] + 0.5) * scale_x - 0.5,
        ],
        [
            0.0,
            intrinsics[1, 1] * scale_y,
            (intrinsics[1, 2] + 0.5) * scale_y - 0.5,
        ],
        [0.0, 0.0, 1.0],
    ])


def save_depth_preview(depth: np.ndarray, path: Path) -> None:
    valid = np.isfinite(depth) & (depth > 0.0)
    preview = np.zeros(depth.shape, dtype=np.uint8)

    if np.any(valid):
        lower, upper = np.quantile(depth[valid], [0.02, 0.98])

        if upper <= lower:
            upper = lower + 1e-6

        normalized = np.clip((depth - lower) / (upper - lower), 0.0, 1.0)
        preview[valid] = np.round((1.0 - normalized[valid]) * 255).astype(
            np.uint8
        )

    Image.fromarray(preview, mode="L").save(path)


def process_frame(
    frame_directory: Path,
    output_root: Path,
    rotation_degrees: int,
    depth_source: str,
    minimum_confidence: int,
) -> dict[str, Any]:
    metadata_path = frame_directory / "metadata.json"
    metadata = load_json(metadata_path)

    frame_id = str(metadata.get("frame_id", frame_directory.name))
    output_directory = output_root / frame_id
    output_directory.mkdir(parents=True, exist_ok=True)

    rgb_width, rgb_height = map(int, metadata["rgb_size"])
    depth_width, depth_height = map(int, metadata["depth_size"])

    rgb_path = frame_directory / metadata.get("rgb_filename", "rgb.png")
    confidence_path = frame_directory / metadata.get(
        "confidence_filename",
        "confidence.bin",
    )

    if depth_source == "smoothed":
        depth_filename = metadata.get(
            "smoothed_depth_filename",
            "depth_smoothed.bin",
        )
    else:
        depth_filename = metadata.get("raw_depth_filename", "depth_raw.bin")

    depth_path = frame_directory / depth_filename

    for required_path in (
        metadata_path,
        rgb_path,
        depth_path,
        confidence_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Required capture file missing: {required_path}")

    with Image.open(rgb_path) as source_image:
        rgb = source_image.convert("RGB")

        if rgb.size != (rgb_width, rgb_height):
            raise ValueError(
                f"{rgb_path} has size {rgb.size}; "
                f"metadata specifies {(rgb_width, rgb_height)}."
            )

        corrected_rgb = rotate_image(rgb, rotation_degrees)
        corrected_rgb.save(output_directory / "rgb.png")

    depth = load_binary_array(
        depth_path,
        np.dtype("<f4"),
        depth_width,
        depth_height,
    )
    confidence = load_binary_array(
        confidence_path,
        np.dtype("u1"),
        depth_width,
        depth_height,
    )

    corrected_depth = rotate_array(depth, rotation_degrees).astype(np.float32)
    corrected_confidence = rotate_array(
        confidence,
        rotation_degrees,
    ).astype(np.uint8)

    valid_depth = np.isfinite(corrected_depth) & (corrected_depth > 0.0)
    trusted_depth = valid_depth & (
        corrected_confidence >= minimum_confidence
    )

    filtered_depth = corrected_depth.copy()
    filtered_depth[~trusted_depth] = np.nan

    np.save(output_directory / "depth.npy", corrected_depth)
    np.save(output_directory / "depth_filtered.npy", filtered_depth)
    np.save(output_directory / "confidence.npy", corrected_confidence)
    save_depth_preview(filtered_depth, output_directory / "depth_preview.png")

    source_intrinsics = np.asarray(
        metadata["camera_intrinsics_row_major"],
        dtype=np.float64,
    )

    corrected_rgb_intrinsics, corrected_rgb_width, corrected_rgb_height = (
        rotate_intrinsics(
            source_intrinsics,
            rgb_width,
            rgb_height,
            rotation_degrees,
        )
    )

    corrected_depth_height, corrected_depth_width = corrected_depth.shape
    corrected_depth_intrinsics = scale_intrinsics(
        corrected_rgb_intrinsics,
        corrected_rgb_width,
        corrected_rgb_height,
        corrected_depth_width,
        corrected_depth_height,
    )

    valid_values = corrected_depth[valid_depth]
    confidence_counts = {
        str(level): int(np.count_nonzero(corrected_confidence == level))
        for level in (0, 1, 2)
    }

    corrected_metadata = {
        "schema_version": 1,
        "frame_id": frame_id,
        "source_directory": str(frame_directory.resolve()),
        "source_metadata": metadata,
        "orientation_correction_degrees_clockwise": rotation_degrees,
        "depth_source": depth_source,
        "depth_measurement_strategy": metadata.get(
            "depth_measurement_strategy",
            "camera_z",
        ),
        "rgb_filename": "rgb.png",
        "depth_filename": "depth.npy",
        "filtered_depth_filename": "depth_filtered.npy",
        "confidence_filename": "confidence.npy",
        "depth_preview_filename": "depth_preview.png",
        "rgb_size": [corrected_rgb_width, corrected_rgb_height],
        "depth_size": [corrected_depth_width, corrected_depth_height],
        "rgb_camera_intrinsics_row_major": (
            corrected_rgb_intrinsics.tolist()
        ),
        "depth_camera_intrinsics_row_major": (
            corrected_depth_intrinsics.tolist()
        ),
        "camera_coordinate_convention": {
            "x": "right in corrected image",
            "y": "up in corrected image",
            "z": "forward from camera",
            "depth_value": "camera-axis Z distance in metres",
        },
        "arkit_camera_transform_row_major": metadata.get(
            "camera_transform_row_major"
        ),
        "minimum_confidence_retained": minimum_confidence,
        "valid_depth_fraction": float(np.mean(valid_depth)),
        "trusted_depth_fraction": float(np.mean(trusted_depth)),
        "depth_min_m": float(np.min(valid_values)),
        "depth_median_m": float(np.median(valid_values)),
        "depth_max_m": float(np.max(valid_values)),
        "confidence_counts": confidence_counts,
    }

    write_json(output_directory / "metadata.json", corrected_metadata)

    return {
        "frame_id": frame_id,
        "output_directory": str(output_directory.resolve()),
        "valid_depth_fraction": float(np.mean(valid_depth)),
        "trusted_depth_fraction": float(np.mean(trusted_depth)),
        "depth_min_m": float(np.min(valid_values)),
        "depth_median_m": float(np.median(valid_values)),
        "depth_max_m": float(np.max(valid_values)),
    }


def adapt_real_rgbd_dataset(
    input_root: Path,
    output_root: Path,
    rotation_degrees: int = 180,
    depth_source: str = "smoothed",
    minimum_confidence: int = 1,
) -> dict[str, Any]:
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input capture directory not found: {input_root}")

    if minimum_confidence not in (0, 1, 2):
        raise ValueError("Minimum confidence must be 0, 1, or 2.")

    frame_directories = sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir() and (path / "metadata.json").exists()
    )

    if not frame_directories:
        raise ValueError(f"No captured frame directories found under {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    frames = [
        process_frame(
            frame_directory=frame_directory,
            output_root=output_root,
            rotation_degrees=rotation_degrees,
            depth_source=depth_source,
            minimum_confidence=minimum_confidence,
        )
        for frame_directory in frame_directories
    ]

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_frames": len(frames),
        "orientation_correction_degrees_clockwise": rotation_degrees,
        "depth_source": depth_source,
        "minimum_confidence_retained": minimum_confidence,
        "frames": frames,
    }

    write_json(output_root / "adapter_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert iPhone ARKit captures into canonical RGB-D data."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--rotation",
        type=int,
        choices=(0, 90, 180, 270),
        default=180,
    )
    parser.add_argument(
        "--depth-source",
        choices=("raw", "smoothed"),
        default="smoothed",
    )
    parser.add_argument(
        "--minimum-confidence",
        type=int,
        choices=(0, 1, 2),
        default=1,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    summary = adapt_real_rgbd_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        rotation_degrees=args.rotation,
        depth_source=args.depth_source,
        minimum_confidence=args.minimum_confidence,
    )

    print(f"Processed {summary['num_frames']} RGB-D frames.")
    print(f"Output: {summary['output_root']}")


if __name__ == "__main__":
    main()
