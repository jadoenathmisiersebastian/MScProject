from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np
from PIL import Image
import yaml

from src.yolo_rgbd_dataset import (
    build_rgbd_array,
    decode_metric_depth,
    encode_metric_depth,
    export_rgbd_segmentation_dataset,
)


def test_metric_depth_encoding_reserves_zero_for_invalid_pixels() -> None:
    depth = np.asarray([[np.nan, 0.3, 1.65, 3.0, 9.0]], dtype=np.float32)
    encoded = encode_metric_depth(depth, depth_min_m=0.3, depth_max_m=3.0)

    assert encoded.tolist() == [[0, 1, 128, 255, 255]]
    decoded = decode_metric_depth(encoded, depth_min_m=0.3, depth_max_m=3.0)
    assert np.isnan(decoded[0, 0])
    assert np.isclose(decoded[0, 1], 0.3)
    assert np.isclose(decoded[0, 3], 3.0)


def test_build_rgbd_array_preserves_rgb_and_adds_depth(tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.npy"
    rgb = np.zeros((6, 10, 3), dtype=np.uint8)
    rgb[:, :, 0] = 20
    rgb[:, :, 1] = 40
    rgb[:, :, 2] = 60
    Image.fromarray(rgb).save(rgb_path)
    np.save(depth_path, np.full((3, 5), 1.0, dtype=np.float32))

    rgbd, valid_fraction = build_rgbd_array(
        rgb_path,
        depth_path,
        long_side=10,
        depth_min_m=0.3,
        depth_max_m=3.0,
    )

    assert rgbd.shape == (6, 10, 4)
    assert rgbd.dtype == np.uint8
    assert np.all(rgbd[:, :, :3] == rgb)
    assert np.all(rgbd[:, :, 3] > 0)
    assert valid_fraction == 1.0


def _write_manifest(path: Path, source_image: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "frame_id",
                "scene_id",
                "source_image_path",
                "is_null_sample",
                "num_target_objects",
            ),
        )
        writer.writeheader()
        writer.writerow({
            "frame_id": "frame_000000",
            "scene_id": "test",
            "source_image_path": source_image,
            "is_null_sample": 0,
            "num_target_objects": 1,
        })


def test_export_rgbd_segmentation_dataset_writes_four_channel_sidecars(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    source_image = raw / "step0.camera.png"
    source_depth = raw / "step0.camera.Depth.npy"
    Image.fromarray(np.full((6, 10, 3), 100, dtype=np.uint8)).save(source_image)
    np.save(source_depth, np.full((3, 5), 1.2, dtype=np.float32))
    (raw / "step0.frame_data.json").write_text(json.dumps({
        "captures": [{
            "filename": source_image.name,
            "annotations": [{
                "@type": "type.unity.com/unity.solo.DepthAnnotation",
                "filename": source_depth.name,
            }],
        }],
    }))

    source_dataset = tmp_path / "source_yolo"
    manifests: list[tuple[str, Path]] = []
    for split in ("train", "val", "test"):
        image_dir = source_dataset / "images" / split
        label_dir = source_dataset / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        Image.open(source_image).save(
            image_dir / "frame_000000_step0.camera.png"
        )
        (label_dir / "frame_000000_step0.camera.txt").write_text(
            "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n"
        )
        manifest = tmp_path / f"{split}_manifest.csv"
        _write_manifest(manifest, source_image)
        manifests.append((split, manifest))
    (source_dataset / "data.yaml").write_text(yaml.safe_dump({
        "path": str(source_dataset),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "object"},
    }, sort_keys=False))

    output = tmp_path / "rgbd_yolo"
    summaries = export_rgbd_segmentation_dataset(
        source_dataset=source_dataset,
        manifest_splits=manifests,
        output_path=output,
        long_side=10,
    )

    assert set(summaries) == {"train", "val", "test"}
    rgbd = np.load(
        output / "images/train/frame_000000_step0.camera.npy",
        allow_pickle=False,
    )
    assert rgbd.shape == (6, 10, 4)
    assert yaml.safe_load((output / "data.yaml").read_text())["channels"] == 4
    assert json.loads((output / "rgbd_encoding.json").read_text())["depth_min_m"] == 0.3
