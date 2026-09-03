from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Any

from PIL import Image

import torch
from torchvision import transforms

from .vision_crop_spatial_baseline import (
    CropSpatialRegressor,
    MetadataNormalizer,
    TARGET_COLUMNS,
    TargetNormalizer,
    build_metadata_tensor,
    prepare_crop_rows,
)


def _read_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(csv_path).expanduser().resolve()

    with csv_path.open("r") as f:
        return list(csv.DictReader(f))


def _write_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No prediction rows to write.")

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def _load_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_encoding = checkpoint.get("class_encoding", "ordinal")
    class_feature_dim = (
        len(checkpoint["class_to_id"])
        if class_encoding == "one_hot"
        else 1
    )

    model = CropSpatialRegressor(
        metadata_dim=len(checkpoint["metadata_columns"]) + class_feature_dim,
        output_dim=len(checkpoint["target_columns"]),
        use_pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    normalizer_data = checkpoint["target_normalizer"]

    target_normalizer = TargetNormalizer(
        mean=torch.tensor(normalizer_data["mean"], dtype=torch.float32),
        std=torch.tensor(normalizer_data["std"], dtype=torch.float32),
    )

    metadata_normalizer = None
    metadata_normalizer_data = checkpoint.get("metadata_normalizer")

    if metadata_normalizer_data is not None:
        metadata_normalizer = MetadataNormalizer(
            mean=torch.tensor(metadata_normalizer_data["mean"], dtype=torch.float32),
            std=torch.tensor(metadata_normalizer_data["std"], dtype=torch.float32),
        )

    return model, checkpoint, target_normalizer, metadata_normalizer


def export_crop_spatial_predictions(
    crop_labels_csv: str | Path,
    checkpoint_path: str | Path,
    output_csv: str | Path,
) -> Path:
    rows = prepare_crop_rows(_read_rows(crop_labels_csv))

    if not rows:
        raise ValueError("No crop rows found.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, checkpoint, target_normalizer, metadata_normalizer = _load_model(
        checkpoint_path,
        device,
    )

    image_size = int(checkpoint["image_size"])
    class_to_id = checkpoint["class_to_id"]
    class_column = checkpoint.get("class_column", "class_name")
    class_encoding = checkpoint.get("class_encoding", "ordinal")
    metadata_columns = checkpoint["metadata_columns"]
    target_columns = checkpoint["target_columns"]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    output_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for row in rows:
            image = Image.open(row["crop_image_path"]).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            metadata_tensor = build_metadata_tensor(
                row=row,
                class_to_id=class_to_id,
                metadata_columns=metadata_columns,
                class_encoding=class_encoding,
                metadata_normalizer=metadata_normalizer,
                class_column=class_column,
            ).unsqueeze(0).to(device)

            prediction_normalized = model(image_tensor, metadata_tensor)
            prediction = target_normalizer.denormalize(prediction_normalized)[0].cpu().tolist()

            output_row = {
                "crop_image_path": row["crop_image_path"],
                "source_image_path": row["source_image_path"],
                "frame_id": row["frame_id"],
                "scene_id": row["scene_id"],
                "class_name": row["class_name"],
                "semantic_class": row.get("semantic_class", row["class_name"]),
                "is_focused_object": row["is_focused_object"],
            }

            for index, column in enumerate(target_columns):
                true_value = float(row[column])
                predicted_value = float(prediction[index])

                output_row[f"true_{column}"] = true_value
                output_row[f"pred_{column}"] = predicted_value
                output_row[f"abs_error_{column}"] = abs(true_value - predicted_value)

            output_rows.append(output_row)

    return _write_csv(output_rows, output_csv)


def summarize_prediction_errors(prediction_csv: str | Path, output_json: str | Path | None = None) -> dict[str, Any]:
    rows = _read_rows(prediction_csv)

    summary: dict[str, Any] = {
        "num_rows": len(rows),
        "mae": {},
    }

    for column in TARGET_COLUMNS:
        key = f"abs_error_{column}"
        errors = [float(row[key]) for row in rows]
        summary["mae"][column] = sum(errors) / len(errors) if errors else 0.0

    if output_json is not None:
        output_json = Path(output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)

        with output_json.open("w") as f:
            json.dump(summary, f, indent=2)

    return summary
