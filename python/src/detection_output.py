from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

import cv2
from ultralytics import YOLO

from .config import ProjectConfig
from .utils import IMAGE_EXTENSIONS, ensure_split


@dataclass(frozen=True)
class Detection:
    detection_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    bbox_xywh: list[float]
    image_center: list[float]
    normalized_center: list[float]
    area_pixels: float


@dataclass(frozen=True)
class FrameDetections:
    frame_id: str
    image_path: str
    image_width: int
    image_height: int
    detections: list[Detection]


def _detection_from_box(box, detection_id: int, names: dict, image_width: int, image_height: int) -> Detection:
    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = names.get(class_id, str(class_id))

    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2
    center_y = y1 + height / 2

    return Detection(
        detection_id=detection_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=[x1, y1, x2, y2],
        bbox_xywh=[center_x, center_y, width, height],
        image_center=[center_x, center_y],
        normalized_center=[center_x / image_width, center_y / image_height],
        area_pixels=width * height,
    )


def run_detection_export(
    config: ProjectConfig,
    split: str = "test",
    weights: str | Path | None = None,
    input_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    conf: float = 0.25,
) -> Path:
    split = ensure_split(split)

    weights_path = Path(weights).expanduser().resolve() if weights else config.model_weights
    source_dir = Path(input_dir).expanduser().resolve() if input_dir else config.yolo_dataset_output / "images" / split

    if output_path is None:
        destination_path = config.predictions_output / split / "detections.json"
    else:
        destination_path = Path(output_path).expanduser().resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Input image folder not found: {source_dir}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))

    image_paths = sorted(
        path for path in source_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    all_frames: list[FrameDetections] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        image_height, image_width = image.shape[:2]
        results = model(image, conf=conf, verbose=False)[0]

        detections = [
            _detection_from_box(
                box=box,
                detection_id=i,
                names=results.names,
                image_width=image_width,
                image_height=image_height,
            )
            for i, box in enumerate(results.boxes)
        ]

        frame = FrameDetections(
            frame_id=image_path.stem,
            image_path=str(image_path),
            image_width=image_width,
            image_height=image_height,
            detections=detections,
        )

        all_frames.append(frame)

    payload = {
        "split": split,
        "weights": str(weights_path),
        "input_dir": str(source_dir),
        "frames": [
            {
                **asdict(frame),
                "detections": [asdict(det) for det in frame.detections],
            }
            for frame in all_frames
        ],
    }

    with open(destination_path, "w") as f:
        json.dump(payload, f, indent=2)

    return destination_path