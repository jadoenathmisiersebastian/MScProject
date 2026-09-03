from __future__ import annotations
from typing import Iterable

from pathlib import Path
import csv

from .vision_label_dataset import load_vision_labels


def _semantic_class(obj: dict) -> str:
    value = obj.get("semantic_class")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "Vision object is missing semantic_class. Run semantic enrichment or regenerate labels with the updated Unity exporter."
        )
    return value


def build_object_spatial_rows(records: list[dict]) -> list[dict]:
    rows: list[dict] = []

    for record in records:
        image_width = record["camera"]["image_width"]
        image_height = record["camera"]["image_height"]

        for obj in record["objects"]:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            bbox_width = max(0.0, x2 - x1)
            bbox_height = max(0.0, y2 - y1)

            position_camera = obj["position_camera"]
            dimensions = obj["dimensions_m"]

            rows.append({
                "frame_id": record["frame_id"],
                "scene_id": record["scene_id"],
                "image_path": record["image_path"],

                "image_width": image_width,
                "image_height": image_height,

                "object_id": obj["object_id"],
                "object_name": obj["object_name"],
                "class_name": obj["class_name"],
                "semantic_class": _semantic_class(obj),
                "is_focused_object": int(obj["is_focused_object"]),
                "is_in_front_of_camera": int(obj["is_in_front_of_camera"]),

                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "bbox_center_x": obj["image_center"][0],
                "bbox_center_y": obj["image_center"][1],
                "bbox_center_x_norm": obj["normalized_center"][0],
                "bbox_center_y_norm": obj["normalized_center"][1],
                "bbox_area_pixels": obj["bbox_area_pixels"],
                "bbox_area_normalized": obj["bbox_area_normalized"],

                "position_camera_x": position_camera[0],
                "position_camera_y": position_camera[1],
                "position_camera_z": position_camera[2],
                "distance_camera_m": obj["distance_camera_m"],

                "dimension_x": dimensions[0],
                "dimension_y": dimensions[1],
                "dimension_z": dimensions[2],
            })

    return rows


def write_object_spatial_csv(rows: list[dict], output_path: str | Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def export_object_spatial_features(
    vision_labels_path: str | Path | Iterable[str | Path],
    output_path: str | Path,
    validate: bool = True,
) -> Path:
    if isinstance(vision_labels_path, (str, Path)):
        paths = [vision_labels_path]
    else:
        paths = list(vision_labels_path)

    records = load_vision_labels(paths, validate=validate)
    rows = build_object_spatial_rows(records)
    return write_object_spatial_csv(rows, output_path)
