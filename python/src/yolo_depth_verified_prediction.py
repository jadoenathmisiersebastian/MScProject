from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .depth_candidate_verifier import (
    DepthVerifierConfig,
    extract_depth_candidate_features,
    verify_depth_candidate,
)


def predict_depth_verified_segmentation(
    weights: str | Path,
    rgb_path: str | Path,
    depth_path: str | Path,
    metadata_path: str | Path,
    proposal_confidence: float = 0.25,
    image_size: int = 640,
    device: str | None = None,
    maximum_focus_distance: float = 0.25,
    verifier_config: DepthVerifierConfig | None = None,
    model: Any | None = None,
) -> dict:
    """Generate RGB proposals and reject only strong metric-depth contradictions."""
    from PIL import Image
    from ultralytics import YOLO

    rgb_path = Path(rgb_path).expanduser().resolve()
    depth_path = Path(depth_path).expanduser().resolve()
    metadata_path = Path(metadata_path).expanduser().resolve()
    for path in (rgb_path, depth_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(path)

    with Image.open(rgb_path) as image:
        rgb_size = image.size
    depth = np.load(depth_path, allow_pickle=False).astype(np.float32, copy=False)
    metadata = json.loads(metadata_path.read_text())
    intrinsics = np.asarray(metadata["depth_camera_intrinsics_row_major"], dtype=np.float64)

    if model is None:
        model = YOLO(str(Path(weights).expanduser().resolve()))
    result = model.predict(
        str(rgb_path),
        conf=proposal_confidence,
        imgsz=image_size,
        device=device,
        verbose=False,
    )[0]
    boxes = result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None else np.empty((0, 4))
    confidences = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.empty((0,))
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes is not None else np.empty((0,), dtype=int)
    polygons = list(result.masks.xy) if result.masks is not None else []

    candidates = []
    for index, (bbox, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
        polygon = np.asarray(polygons[index]) if index < len(polygons) else None
        features = extract_depth_candidate_features(
            depth_m=depth,
            rgb_size=rgb_size,
            bbox_xyxy=bbox,
            polygon_xy=polygon,
            depth_intrinsics=intrinsics,
            ring_radius_pixels=(verifier_config or DepthVerifierConfig()).ring_radius_pixels,
        )
        verification = verify_depth_candidate(features, verifier_config)
        center_x = float((bbox[0] + bbox[2]) * 0.5)
        center_y = float((bbox[1] + bbox[3]) * 0.5)
        center_distance = math.hypot(
            (center_x - rgb_size[0] * 0.5) / rgb_size[0],
            (center_y - rgb_size[1] * 0.5) / rgb_size[1],
        )
        candidates.append(
            {
                "prediction_index": index,
                "class_id": int(class_id),
                "class_name": str(model.names[int(class_id)]),
                "confidence": float(confidence),
                "bbox_xyxy": [float(value) for value in bbox],
                "normalized_distance_from_image_center": center_distance,
                "inside_focus_region": center_distance <= maximum_focus_distance,
                "depth_verification": verification.to_dict(),
            }
        )

    accepted = [
        candidate
        for candidate in candidates
        if candidate["inside_focus_region"]
        and candidate["depth_verification"]["accepted"]
    ]
    focused = max(
        accepted,
        key=lambda candidate: (
            candidate["confidence"],
            -candidate["normalized_distance_from_image_center"],
        ),
        default=None,
    )
    return {
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "proposal_confidence": proposal_confidence,
        "maximum_focus_distance": maximum_focus_distance,
        "candidates": candidates,
        "focused_candidate": focused,
        "result": result,
    }
