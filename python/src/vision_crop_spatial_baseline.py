from __future__ import annotations

from pathlib import Path
import csv
import json
import random
from typing import Any

from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


TARGET_COLUMNS = [
    "distance_camera_m",
    "position_camera_x",
    "position_camera_y",
    "position_camera_z",
    "dimension_x",
    "dimension_y",
    "dimension_z",
]

MODEL_VERSION = 2

LEGACY_METADATA_COLUMNS = [
    "bbox_area_normalized",
    "crop_width",
    "crop_height",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
]

METADATA_COLUMNS = [
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_center_x_norm",
    "bbox_center_y_norm",
    "bbox_area_normalized",
    "bbox_aspect_ratio",
]
CLASS_COLUMN = "semantic_class"
CLASS_ENCODING = "one_hot"


class TargetNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = torch.clamp(std, min=1e-6)

    @classmethod
    def from_targets(cls, targets: list[list[float]]) -> "TargetNormalizer":
        tensor = torch.tensor(targets, dtype=torch.float32)
        return cls(
            mean=tensor.mean(dim=0),
            std=tensor.std(dim=0),
        )

    def normalize(self, target: torch.Tensor) -> torch.Tensor:
        return (target - self.mean) / self.std

    def denormalize(self, target: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(target.device)
        std = self.std.to(target.device)
        return target * std + mean

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


class MetadataNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = torch.clamp(std, min=1e-6)

    @classmethod
    def from_rows(
        cls,
        rows: list[dict[str, Any]],
        columns: list[str],
    ) -> "MetadataNormalizer":
        values = [
            [float(row[column]) for column in columns]
            for row in rows
        ]
        tensor = torch.tensor(values, dtype=torch.float32)
        return cls(
            mean=tensor.mean(dim=0),
            std=tensor.std(dim=0),
        )

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


def prepare_crop_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    image_size_cache: dict[str, tuple[int, int]] = {}

    for row in rows:
        if all(str(row.get(column, "")).strip() for column in METADATA_COLUMNS):
            continue

        width_value = row.get("source_image_width")
        height_value = row.get("source_image_height")

        if width_value and height_value:
            image_width = int(float(width_value))
            image_height = int(float(height_value))
        else:
            source_path = row.get("source_image_path")

            if not source_path:
                raise ValueError(
                    "Crop row is missing source image dimensions and source_image_path."
                )

            if source_path not in image_size_cache:
                with Image.open(source_path) as image:
                    image_size_cache[source_path] = image.size

            image_width, image_height = image_size_cache[source_path]

        if image_width <= 0 or image_height <= 0:
            raise ValueError("Source image dimensions must be positive.")

        x1 = float(row["bbox_x1"])
        y1 = float(row["bbox_y1"])
        x2 = float(row["bbox_x2"])
        y2 = float(row["bbox_y2"])
        bbox_width = max(0.0, x2 - x1)
        bbox_height = max(0.0, y2 - y1)

        row["source_image_width"] = str(image_width)
        row["source_image_height"] = str(image_height)
        row["bbox_width_norm"] = str(bbox_width / image_width)
        row["bbox_height_norm"] = str(bbox_height / image_height)
        row["bbox_center_x_norm"] = str((x1 + bbox_width / 2.0) / image_width)
        row["bbox_center_y_norm"] = str((y1 + bbox_height / 2.0) / image_height)
        row["bbox_aspect_ratio"] = str(
            bbox_width / bbox_height if bbox_height > 0.0 else 0.0
        )

    return rows


def build_metadata_tensor(
    row: dict[str, Any],
    class_to_id: dict[str, int],
    metadata_columns: list[str],
    class_encoding: str,
    metadata_normalizer: MetadataNormalizer | None = None,
    class_column: str = CLASS_COLUMN,
) -> torch.Tensor:
    continuous = torch.tensor(
        [float(row[column]) for column in metadata_columns],
        dtype=torch.float32,
    )

    if metadata_normalizer is not None:
        continuous = metadata_normalizer.normalize(continuous)

    class_name = row[class_column]

    if class_name not in class_to_id:
        raise ValueError(f"Unknown semantic class in crop data: {class_name}")

    class_id = class_to_id[class_name]

    if class_encoding == "one_hot":
        class_features = torch.zeros(len(class_to_id), dtype=torch.float32)
        class_features[class_id] = 1.0
    elif class_encoding == "ordinal":
        class_features = torch.tensor([float(class_id)], dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported class encoding: {class_encoding}")

    return torch.cat([continuous, class_features])


class VisionCropSpatialDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        class_to_id: dict[str, int],
        target_normalizer: TargetNormalizer,
        metadata_normalizer: MetadataNormalizer,
        image_size: int = 128,
    ):
        self.rows = rows
        self.class_to_id = class_to_id
        self.target_normalizer = target_normalizer
        self.metadata_normalizer = metadata_normalizer

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]

        image = Image.open(row["crop_image_path"]).convert("RGB")
        image_tensor = self.transform(image)

        metadata_tensor = build_metadata_tensor(
            row=row,
            class_to_id=self.class_to_id,
            metadata_columns=METADATA_COLUMNS,
            class_encoding=CLASS_ENCODING,
            metadata_normalizer=self.metadata_normalizer,
        )

        target = torch.tensor(
            [float(row[column]) for column in TARGET_COLUMNS],
            dtype=torch.float32,
        )

        target_tensor = self.target_normalizer.normalize(target)

        return image_tensor, metadata_tensor, target_tensor


class CropSpatialRegressor(nn.Module):
    def __init__(self, metadata_dim: int, output_dim: int, use_pretrained: bool = True):
        super().__init__()

        if use_pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
                backbone = models.resnet18(weights=weights)
            except Exception as exc:
                print(f"Could not load pretrained ResNet18 weights, falling back to random init: {exc}")
                backbone = models.resnet18(weights=None)
        else:
            backbone = models.resnet18(weights=None)

        image_feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Linear(image_feature_dim + metadata_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, image, metadata):
        image_features = self.backbone(image)
        features = torch.cat([image_features, metadata], dim=1)
        return self.head(features)


def _read_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(csv_path).expanduser().resolve()

    with csv_path.open("r") as f:
        rows = list(csv.DictReader(f))

    return prepare_crop_rows(rows)


def _class_mapping(rows: list[dict[str, Any]]) -> dict[str, int]:
    classes = sorted({row[CLASS_COLUMN] for row in rows})
    return {name: index for index, name in enumerate(classes)}


def _validate_classes(rows: list[dict[str, Any]], class_to_id: dict[str, int], split_name: str) -> None:
    unknown = sorted({row[CLASS_COLUMN] for row in rows} - set(class_to_id))

    if unknown:
        raise ValueError(
            f"{split_name} contains semantic classes absent from training: {unknown}"
        )


def _train_test_split(
    rows: list[dict[str, Any]],
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    test_count = max(1, int(len(shuffled) * test_ratio))
    return shuffled[test_count:], shuffled[:test_count]


def _targets_from_rows(rows: list[dict[str, Any]]) -> list[list[float]]:
    return [
        [float(row[column]) for column in TARGET_COLUMNS]
        for row in rows
    ]


def _evaluate(model, dataloader, device, target_normalizer: TargetNormalizer) -> dict[str, float]:
    model.eval()

    absolute_errors = []
    euclidean_position_errors = []
    euclidean_dimension_errors = []

    with torch.no_grad():
        for images, metadata, targets_normalized in dataloader:
            images = images.to(device)
            metadata = metadata.to(device)
            targets_normalized = targets_normalized.to(device)

            predictions_normalized = model(images, metadata)

            predictions = target_normalizer.denormalize(predictions_normalized)
            targets = target_normalizer.denormalize(targets_normalized)

            errors = torch.abs(predictions - targets)
            absolute_errors.append(errors.cpu())

            position_error = torch.sqrt(
                torch.sum(
                    (predictions[:, 1:4] - targets[:, 1:4]) ** 2,
                    dim=1,
                )
            )

            dimension_error = torch.sqrt(
                torch.sum(
                    (predictions[:, 4:7] - targets[:, 4:7]) ** 2,
                    dim=1,
                )
            )

            euclidean_position_errors.append(position_error.cpu())
            euclidean_dimension_errors.append(dimension_error.cpu())

    all_errors = torch.cat(absolute_errors, dim=0)
    all_position_errors = torch.cat(euclidean_position_errors, dim=0)
    all_dimension_errors = torch.cat(euclidean_dimension_errors, dim=0)

    metrics = {}

    for index, column in enumerate(TARGET_COLUMNS):
        metrics[f"mae_{column}"] = float(all_errors[:, index].mean())

    metrics["mean_position_euclidean_error_m"] = float(all_position_errors.mean())
    metrics["mean_dimension_euclidean_error_m"] = float(all_dimension_errors.mean())

    return metrics


def train_crop_spatial_baseline(
    crop_labels_csv: str | Path,
    output_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 16,
    image_size: int = 128,
    test_ratio: float = 0.2,
    seed: int = 42,
    use_pretrained: bool = True,
) -> dict[str, Any]:
    rows = _read_rows(crop_labels_csv)

    if len(rows) < 10:
        raise ValueError("Need at least 10 crop rows for training.")

    random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows, test_rows = _train_test_split(rows, test_ratio=test_ratio, seed=seed)
    class_to_id = _class_mapping(train_rows)
    _validate_classes(test_rows, class_to_id, "Test split")

    target_normalizer = TargetNormalizer.from_targets(_targets_from_rows(train_rows))
    metadata_normalizer = MetadataNormalizer.from_rows(train_rows, METADATA_COLUMNS)

    train_dataset = VisionCropSpatialDataset(
        train_rows,
        class_to_id,
        target_normalizer=target_normalizer,
        metadata_normalizer=metadata_normalizer,
        image_size=image_size,
    )

    test_dataset = VisionCropSpatialDataset(
        test_rows,
        class_to_id,
        target_normalizer=target_normalizer,
        metadata_normalizer=metadata_normalizer,
        image_size=image_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = CropSpatialRegressor(
        metadata_dim=len(METADATA_COLUMNS) + len(class_to_id),
        output_dim=len(TARGET_COLUMNS),
        use_pretrained=use_pretrained,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, metadata, targets in train_loader:
            images = images.to(device)
            metadata = metadata.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(images, metadata)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        metrics = _evaluate(model, test_loader, device, target_normalizer)
        train_loss = total_loss / max(1, len(train_loader))

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **metrics,
        })

        print(
            f"epoch={epoch + 1}/{epochs} "
            f"train_loss={train_loss:.6f} "
            f"mae_distance={metrics['mae_distance_camera_m']:.4f} "
            f"pos_err={metrics['mean_position_euclidean_error_m']:.4f} "
            f"dim_err={metrics['mean_dimension_euclidean_error_m']:.4f}"
        )

    final_metrics = _evaluate(model, test_loader, device, target_normalizer)

    model_path = output_dir / "crop_spatial_baseline.pt"

    torch.save(
        {
            "model_version": MODEL_VERSION,
            "model_state_dict": model.state_dict(),
            "class_to_id": class_to_id,
            "class_column": CLASS_COLUMN,
            "class_encoding": CLASS_ENCODING,
            "target_columns": TARGET_COLUMNS,
            "metadata_columns": METADATA_COLUMNS,
            "metadata_normalizer": metadata_normalizer.to_dict(),
            "target_normalizer": target_normalizer.to_dict(),
            "image_size": image_size,
            "use_pretrained": use_pretrained,
            "metrics": final_metrics,
        },
        model_path,
    )

    summary = {
        "model_version": MODEL_VERSION,
        "num_rows": len(rows),
        "num_train": len(train_rows),
        "num_test": len(test_rows),
        "class_to_id": class_to_id,
        "class_column": CLASS_COLUMN,
        "class_encoding": CLASS_ENCODING,
        "target_columns": TARGET_COLUMNS,
        "metadata_columns": METADATA_COLUMNS,
        "metadata_normalizer": metadata_normalizer.to_dict(),
        "model_path": str(model_path),
        "use_pretrained": use_pretrained,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "target_normalizer": target_normalizer.to_dict(),
        "metrics": final_metrics,
        "history": history,
    }

    summary_path = output_dir / "summary.json"

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    summary["summary_path"] = str(summary_path)

    return summary


def train_crop_spatial_baseline_from_splits(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    output_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 16,
    image_size: int = 128,
    seed: int = 42,
    use_pretrained: bool = True,
) -> dict[str, Any]:
    train_rows = _read_rows(train_csv)
    val_rows = _read_rows(val_csv)
    test_rows = _read_rows(test_csv)

    if len(train_rows) < 10:
        raise ValueError("Need at least 10 train crop rows.")
    if len(val_rows) < 1:
        raise ValueError("Need at least 1 validation crop row.")
    if len(test_rows) < 1:
        raise ValueError("Need at least 1 test crop row.")

    random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_to_id = _class_mapping(train_rows)
    _validate_classes(val_rows, class_to_id, "Validation split")
    _validate_classes(test_rows, class_to_id, "Test split")

    target_normalizer = TargetNormalizer.from_targets(_targets_from_rows(train_rows))
    metadata_normalizer = MetadataNormalizer.from_rows(train_rows, METADATA_COLUMNS)

    train_dataset = VisionCropSpatialDataset(
        train_rows,
        class_to_id,
        target_normalizer=target_normalizer,
        metadata_normalizer=metadata_normalizer,
        image_size=image_size,
    )

    val_dataset = VisionCropSpatialDataset(
        val_rows,
        class_to_id,
        target_normalizer=target_normalizer,
        metadata_normalizer=metadata_normalizer,
        image_size=image_size,
    )

    test_dataset = VisionCropSpatialDataset(
        test_rows,
        class_to_id,
        target_normalizer=target_normalizer,
        metadata_normalizer=metadata_normalizer,
        image_size=image_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = CropSpatialRegressor(
        metadata_dim=len(METADATA_COLUMNS) + len(class_to_id),
        output_dim=len(TARGET_COLUMNS),
        use_pretrained=use_pretrained,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    history = []
    best_val_position_error = float("inf")
    best_model_path = output_dir / "crop_spatial_baseline_best.pt"

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, metadata, targets in train_loader:
            images = images.to(device)
            metadata = metadata.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(images, metadata)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        train_loss = total_loss / max(1, len(train_loader))
        val_metrics = _evaluate(model, val_loader, device, target_normalizer)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_metrics": val_metrics,
        })

        val_position_error = val_metrics["mean_position_euclidean_error_m"]

        if val_position_error < best_val_position_error:
            best_val_position_error = val_position_error

            torch.save(
                {
                    "model_version": MODEL_VERSION,
                    "model_state_dict": model.state_dict(),
                    "class_to_id": class_to_id,
                    "class_column": CLASS_COLUMN,
                    "class_encoding": CLASS_ENCODING,
                    "target_columns": TARGET_COLUMNS,
                    "metadata_columns": METADATA_COLUMNS,
                    "metadata_normalizer": metadata_normalizer.to_dict(),
                    "target_normalizer": target_normalizer.to_dict(),
                    "image_size": image_size,
                    "use_pretrained": use_pretrained,
                    "val_metrics": val_metrics,
                },
                best_model_path,
            )

        print(
            f"epoch={epoch + 1}/{epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_distance={val_metrics['mae_distance_camera_m']:.4f} "
            f"val_pos_err={val_metrics['mean_position_euclidean_error_m']:.4f} "
            f"val_dim_err={val_metrics['mean_dimension_euclidean_error_m']:.4f}"
        )

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = _evaluate(model, test_loader, device, target_normalizer)

    final_model_path = output_dir / "crop_spatial_baseline_final.pt"

    torch.save(
        {
            "model_version": MODEL_VERSION,
            "model_state_dict": model.state_dict(),
            "class_to_id": class_to_id,
            "class_column": CLASS_COLUMN,
            "class_encoding": CLASS_ENCODING,
            "target_columns": TARGET_COLUMNS,
            "metadata_columns": METADATA_COLUMNS,
            "metadata_normalizer": metadata_normalizer.to_dict(),
            "target_normalizer": target_normalizer.to_dict(),
            "image_size": image_size,
            "use_pretrained": use_pretrained,
            "test_metrics": test_metrics,
        },
        final_model_path,
    )

    summary = {
        "model_version": MODEL_VERSION,
        "split_mode": "explicit_train_val_test",
        "train_csv": str(Path(train_csv).expanduser().resolve()),
        "val_csv": str(Path(val_csv).expanduser().resolve()),
        "test_csv": str(Path(test_csv).expanduser().resolve()),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_test": len(test_rows),
        "class_to_id": class_to_id,
        "class_column": CLASS_COLUMN,
        "class_encoding": CLASS_ENCODING,
        "target_columns": TARGET_COLUMNS,
        "metadata_columns": METADATA_COLUMNS,
        "metadata_normalizer": metadata_normalizer.to_dict(),
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "use_pretrained": use_pretrained,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "target_normalizer": target_normalizer.to_dict(),
        "best_val_position_error_m": best_val_position_error,
        "test_metrics": test_metrics,
        "history": history,
    }

    summary_path = output_dir / "summary.json"

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    summary["summary_path"] = str(summary_path)

    return summary
