from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO

from .config import ProjectConfig
from .utils import IMAGE_EXTENSIONS, ensure_split


CLASS_COLORS = {
    0: (0, 0, 255),
    1: (255, 255, 255),
    2: (0, 255, 0),
    3: (0, 255, 255),
    4: (255, 0, 255),
    5: (255, 255, 0),
    6: (0, 128, 255),
    7: (128, 0, 255),
    8: (255, 128, 0),
    9: (255, 0, 0),
}


def _draw_predictions(image, results) -> None:
    detected_classes = set()

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        detected_classes.add(cls)

        colour = CLASS_COLORS.get(cls, (255, 255, 255))
        class_name = results.names.get(cls, str(cls))
        label = f"{class_name} {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )

    legend_x = 15
    legend_y = 25
    for cls in sorted(detected_classes):
        colour = CLASS_COLORS.get(cls, (255, 255, 255))
        class_name = results.names.get(cls, str(cls))
        cv2.rectangle(image, (legend_x, legend_y - 9), (legend_x + 12, legend_y + 3), colour, -1)
        cv2.putText(
            image,
            f"{cls}: {class_name}",
            (legend_x + 20, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        legend_y += 18


def predict_yolo(config: ProjectConfig, split: str = "test", weights: str | Path | None = None, input_dir: str | Path | None = None, output_dir: str | Path | None = None, conf: float = 0.25) -> Path:
    split = ensure_split(split)
    weights_path = Path(weights).expanduser().resolve() if weights else config.model_weights
    source_dir = Path(input_dir).expanduser().resolve() if input_dir else config.yolo_dataset_output / "images" / split
    destination_dir = Path(output_dir).expanduser().resolve() if output_dir else config.predictions_output / split

    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Input image folder not found: {source_dir}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights_path))

    image_paths = sorted(path for path in source_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {source_dir}")

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read image: {image_path}")
            continue
        results = model(image, conf=conf, verbose=False)[0]
        _draw_predictions(image, results)
        cv2.imwrite(str(destination_dir / image_path.name), image)

    return destination_dir
