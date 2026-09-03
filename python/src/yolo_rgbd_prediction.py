from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .yolo_rgbd_dataset import build_rgbd_array


def load_rgbd_input(
    rgb_path: str | Path,
    depth_path: str | Path,
    long_side: int = 640,
    depth_min_m: float = 0.3,
    depth_max_m: float = 3.0,
) -> np.ndarray:
    rgbd, _ = build_rgbd_array(
        rgb_path=rgb_path,
        depth_path=depth_path,
        long_side=long_side,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    return rgbd


def predict_rgbd_segmentation(
    weights: str | Path,
    rgb_path: str | Path,
    depth_path: str | Path,
    image_size: int = 640,
    confidence: float = 0.25,
    device: str | None = None,
    depth_min_m: float = 0.3,
    depth_max_m: float = 3.0,
    **predict_kwargs: Any,
):
    """Run a four-channel YOLO segmentation checkpoint on aligned RGB and depth."""
    from ultralytics import YOLO

    weights_path = Path(weights).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"RGB-D YOLO weights do not exist: {weights_path}")
    rgbd = load_rgbd_input(
        rgb_path=rgb_path,
        depth_path=depth_path,
        long_side=image_size,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    model = YOLO(str(weights_path))
    channels = int(model.model.yaml.get("channels", 3))
    if channels != 4:
        raise ValueError(
            f"Expected a four-channel checkpoint, but model declares {channels} channels."
        )
    return model.predict(
        source=rgbd,
        imgsz=image_size,
        conf=confidence,
        device=device,
        verbose=False,
        **predict_kwargs,
    )[0]
