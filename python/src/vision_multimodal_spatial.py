from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import random
from typing import Any

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .vision_crop_spatial_baseline import (
    CLASS_COLUMN,
    CLASS_ENCODING,
    METADATA_COLUMNS,
    TARGET_COLUMNS,
    MetadataNormalizer,
    TargetNormalizer,
    build_metadata_tensor,
    prepare_crop_rows,
)


RGB_DUAL_CONTEXT = "rgb_dual_context"
DEPTH_ONLY = "depth_only"
RGBD_DUAL_CONTEXT = "rgbd_dual_context"

SPATIAL_ARCHITECTURES = (
    RGB_DUAL_CONTEXT,
    DEPTH_ONLY,
    RGBD_DUAL_CONTEXT,
)

MODEL_FAMILY = "multimodal_spatial"
MODEL_VERSION = 1


def _uses_rgb_crop(architecture: str) -> bool:
    return architecture in {RGB_DUAL_CONTEXT, RGBD_DUAL_CONTEXT}


def _uses_rgb_context(architecture: str) -> bool:
    return architecture in {RGB_DUAL_CONTEXT, RGBD_DUAL_CONTEXT}


def _uses_depth(architecture: str) -> bool:
    return architecture in {DEPTH_ONLY, RGBD_DUAL_CONTEXT}


def _validate_architecture(architecture: str) -> None:
    if architecture not in SPATIAL_ARCHITECTURES:
        raise ValueError(
            f"Unsupported spatial architecture '{architecture}'. "
            f"Choose from: {', '.join(SPATIAL_ARCHITECTURES)}"
        )


class ResizeWithPadding:
    def __init__(self, size: int, fill: tuple[int, int, int] = (124, 116, 104)):
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError("RGB context image dimensions must be positive.")

        scale = min(self.size / width, self.size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BILINEAR,
        )

        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        left = (self.size - resized_width) // 2
        top = (self.size - resized_height) // 2
        canvas.paste(resized, (left, top))
        return canvas


def _rgb_crop_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _rgb_context_transform(image_size: int):
    return transforms.Compose([
        ResizeWithPadding(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _load_depth_tensor(
    path: str | Path,
    image_size: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> torch.Tensor:
    if maximum_depth_m <= minimum_depth_m:
        raise ValueError("maximum_depth_m must be greater than minimum_depth_m.")

    depth = np.load(Path(path).expanduser().resolve(), allow_pickle=False)
    depth = np.asarray(depth, dtype=np.float32)

    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    elif depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]

    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D metric depth crop, got shape {depth.shape}: {path}")

    valid = np.isfinite(depth) & (depth > 0.0)
    clipped = np.clip(depth, minimum_depth_m, maximum_depth_m)
    normalized = (clipped - minimum_depth_m) / (
        maximum_depth_m - minimum_depth_m
    )
    normalized[~valid] = 0.0

    depth_tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    valid_tensor = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    weighted_depth = F.interpolate(
        depth_tensor * valid_tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    interpolated_validity = F.interpolate(
        valid_tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    resized_depth = weighted_depth / torch.clamp(interpolated_validity, min=1e-6)
    resized_mask = (interpolated_validity >= 0.5).to(torch.float32)
    resized_depth = resized_depth * resized_mask

    return torch.cat([resized_depth, resized_mask], dim=1)[0]


def _read_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(csv_path).expanduser().resolve()

    with csv_path.open("r", newline="") as file:
        rows = list(csv.DictReader(file))

    return prepare_crop_rows(rows)


def _class_mapping(rows: list[dict[str, Any]]) -> dict[str, int]:
    classes = sorted({row[CLASS_COLUMN] for row in rows})
    return {name: index for index, name in enumerate(classes)}


def _validate_split(
    rows: list[dict[str, Any]],
    class_to_id: dict[str, int],
    split_name: str,
    architecture: str,
) -> None:
    unknown = sorted({row[CLASS_COLUMN] for row in rows} - set(class_to_id))

    if unknown:
        raise ValueError(
            f"{split_name} contains semantic classes absent from training: {unknown}"
        )

    required_paths = []

    if _uses_rgb_crop(architecture):
        required_paths.append("crop_image_path")
    if _uses_rgb_context(architecture):
        required_paths.append("source_image_path")
    if _uses_depth(architecture):
        required_paths.append("depth_crop_path")

    for row_index, row in enumerate(rows):
        for column in required_paths:
            value = str(row.get(column, "")).strip()

            if not value:
                raise ValueError(
                    f"{split_name} row {row_index} is missing required {column} "
                    f"for architecture {architecture}."
                )

            if not Path(value).expanduser().exists():
                raise FileNotFoundError(
                    f"{split_name} row {row_index} {column} does not exist: {value}"
                )


def _targets_from_rows(rows: list[dict[str, Any]]) -> list[list[float]]:
    return [
        [float(row[column]) for column in TARGET_COLUMNS]
        for row in rows
    ]


class MultimodalSpatialDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        architecture: str,
        class_to_id: dict[str, int],
        target_normalizer: TargetNormalizer,
        metadata_normalizer: MetadataNormalizer,
        image_size: int,
        context_image_size: int,
        minimum_depth_m: float,
        maximum_depth_m: float,
    ):
        _validate_architecture(architecture)
        self.rows = rows
        self.architecture = architecture
        self.class_to_id = class_to_id
        self.target_normalizer = target_normalizer
        self.metadata_normalizer = metadata_normalizer
        self.image_size = image_size
        self.minimum_depth_m = minimum_depth_m
        self.maximum_depth_m = maximum_depth_m
        self.crop_transform = _rgb_crop_transform(image_size)
        self.context_transform = _rgb_context_transform(context_image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        sample: dict[str, torch.Tensor] = {
            "index": torch.tensor(index, dtype=torch.long),
            "metadata": build_metadata_tensor(
                row=row,
                class_to_id=self.class_to_id,
                metadata_columns=METADATA_COLUMNS,
                class_encoding=CLASS_ENCODING,
                metadata_normalizer=self.metadata_normalizer,
            ),
        }

        if _uses_rgb_crop(self.architecture):
            with Image.open(row["crop_image_path"]) as image:
                sample["rgb_crop"] = self.crop_transform(image.convert("RGB"))

        if _uses_rgb_context(self.architecture):
            with Image.open(row["source_image_path"]) as image:
                sample["rgb_context"] = self.context_transform(image.convert("RGB"))

        if _uses_depth(self.architecture):
            sample["depth_crop"] = _load_depth_tensor(
                path=row["depth_crop_path"],
                image_size=self.image_size,
                minimum_depth_m=self.minimum_depth_m,
                maximum_depth_m=self.maximum_depth_m,
            )

        target = torch.tensor(
            [float(row[column]) for column in TARGET_COLUMNS],
            dtype=torch.float32,
        )
        sample["target"] = self.target_normalizer.normalize(target)
        return sample


def _resnet18_encoder(use_pretrained: bool) -> tuple[nn.Module, int]:
    if use_pretrained:
        try:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception as exc:
            print(
                "Could not load pretrained ResNet18 weights, falling back to "
                f"random initialization: {exc}"
            )
            backbone = models.resnet18(weights=None)
    else:
        backbone = models.resnet18(weights=None)

    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim


class DepthEncoder(nn.Module):
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, output_dim),
            nn.ReLU(),
        )
        self.output_dim = output_dim

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.network(depth)


class MultimodalSpatialRegressor(nn.Module):
    def __init__(
        self,
        architecture: str,
        metadata_dim: int,
        output_dim: int,
        use_pretrained: bool = True,
    ):
        super().__init__()
        _validate_architecture(architecture)
        self.architecture = architecture
        feature_dim = metadata_dim

        self.rgb_crop_encoder: nn.Module | None = None
        self.rgb_context_encoder: nn.Module | None = None
        self.depth_encoder: DepthEncoder | None = None

        if _uses_rgb_crop(architecture):
            self.rgb_crop_encoder, crop_dim = _resnet18_encoder(use_pretrained)
            feature_dim += crop_dim

        if _uses_rgb_context(architecture):
            self.rgb_context_encoder, context_dim = _resnet18_encoder(use_pretrained)
            feature_dim += context_dim

        if _uses_depth(architecture):
            self.depth_encoder = DepthEncoder(output_dim=128)
            feature_dim += self.depth_encoder.output_dim

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(256, output_dim),
        )

    def forward(
        self,
        metadata: torch.Tensor,
        rgb_crop: torch.Tensor | None = None,
        rgb_context: torch.Tensor | None = None,
        depth_crop: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = [metadata]

        if self.rgb_crop_encoder is not None:
            if rgb_crop is None:
                raise ValueError("rgb_crop is required by this architecture.")
            features.append(self.rgb_crop_encoder(rgb_crop))

        if self.rgb_context_encoder is not None:
            if rgb_context is None:
                raise ValueError("rgb_context is required by this architecture.")
            features.append(self.rgb_context_encoder(rgb_context))

        if self.depth_encoder is not None:
            if depth_crop is None:
                raise ValueError("depth_crop is required by this architecture.")
            features.append(self.depth_encoder(depth_crop))

        return self.head(torch.cat(features, dim=1))


def _model_inputs(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    return {
        "metadata": batch["metadata"].to(device),
        "rgb_crop": batch.get("rgb_crop").to(device) if "rgb_crop" in batch else None,
        "rgb_context": batch.get("rgb_context").to(device) if "rgb_context" in batch else None,
        "depth_crop": batch.get("depth_crop").to(device) if "depth_crop" in batch else None,
    }


def _evaluate(
    model: MultimodalSpatialRegressor,
    dataloader: DataLoader,
    device: torch.device,
    target_normalizer: TargetNormalizer,
) -> dict[str, float]:
    model.eval()
    absolute_errors = []
    position_errors = []
    dimension_errors = []

    with torch.no_grad():
        for batch in dataloader:
            targets_normalized = batch["target"].to(device)
            predictions_normalized = model(**_model_inputs(batch, device))
            predictions = target_normalizer.denormalize(predictions_normalized)
            targets = target_normalizer.denormalize(targets_normalized)

            absolute_errors.append(torch.abs(predictions - targets).cpu())
            position_errors.append(
                torch.sqrt(torch.sum((predictions[:, 1:4] - targets[:, 1:4]) ** 2, dim=1)).cpu()
            )
            dimension_errors.append(
                torch.sqrt(torch.sum((predictions[:, 4:7] - targets[:, 4:7]) ** 2, dim=1)).cpu()
            )

    all_errors = torch.cat(absolute_errors, dim=0)
    metrics = {
        f"mae_{column}": float(all_errors[:, index].mean())
        for index, column in enumerate(TARGET_COLUMNS)
    }
    metrics["mean_position_euclidean_error_m"] = float(
        torch.cat(position_errors).mean()
    )
    metrics["mean_dimension_euclidean_error_m"] = float(
        torch.cat(dimension_errors).mean()
    )
    return metrics


def _normalizer_from_dict(
    normalizer_type,
    data: dict[str, list[float]],
):
    return normalizer_type(
        mean=torch.tensor(data["mean"], dtype=torch.float32),
        std=torch.tensor(data["std"], dtype=torch.float32),
    )


def _checkpoint_payload(
    model: MultimodalSpatialRegressor,
    architecture: str,
    class_to_id: dict[str, int],
    target_normalizer: TargetNormalizer,
    metadata_normalizer: MetadataNormalizer,
    image_size: int,
    context_image_size: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
    use_pretrained: bool,
) -> dict[str, Any]:
    return {
        "model_family": MODEL_FAMILY,
        "model_version": MODEL_VERSION,
        "architecture": architecture,
        "model_state_dict": model.state_dict(),
        "class_to_id": class_to_id,
        "class_column": CLASS_COLUMN,
        "class_encoding": CLASS_ENCODING,
        "target_columns": TARGET_COLUMNS,
        "metadata_columns": METADATA_COLUMNS,
        "metadata_normalizer": metadata_normalizer.to_dict(),
        "target_normalizer": target_normalizer.to_dict(),
        "image_size": image_size,
        "context_image_size": context_image_size,
        "minimum_depth_m": minimum_depth_m,
        "maximum_depth_m": maximum_depth_m,
        "use_pretrained": use_pretrained,
    }


def train_multimodal_spatial_from_splits(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    output_dir: str | Path,
    architecture: str,
    epochs: int = 50,
    batch_size: int = 8,
    image_size: int = 128,
    context_image_size: int = 128,
    minimum_depth_m: float = 0.3,
    maximum_depth_m: float = 3.0,
    seed: int = 42,
    use_pretrained: bool = True,
) -> dict[str, Any]:
    _validate_architecture(architecture)
    train_rows = _read_rows(train_csv)
    val_rows = _read_rows(val_csv)
    test_rows = _read_rows(test_csv)

    if len(train_rows) < 10:
        raise ValueError("Need at least 10 training rows.")
    if not val_rows or not test_rows:
        raise ValueError("Validation and test splits must not be empty.")

    random.seed(seed)
    torch.manual_seed(seed)

    class_to_id = _class_mapping(train_rows)
    _validate_split(train_rows, class_to_id, "Training split", architecture)
    _validate_split(val_rows, class_to_id, "Validation split", architecture)
    _validate_split(test_rows, class_to_id, "Test split", architecture)

    target_normalizer = TargetNormalizer.from_targets(_targets_from_rows(train_rows))
    metadata_normalizer = MetadataNormalizer.from_rows(train_rows, METADATA_COLUMNS)

    dataset_kwargs = {
        "architecture": architecture,
        "class_to_id": class_to_id,
        "target_normalizer": target_normalizer,
        "metadata_normalizer": metadata_normalizer,
        "image_size": image_size,
        "context_image_size": context_image_size,
        "minimum_depth_m": minimum_depth_m,
        "maximum_depth_m": maximum_depth_m,
    }
    train_dataset = MultimodalSpatialDataset(train_rows, **dataset_kwargs)
    val_dataset = MultimodalSpatialDataset(val_rows, **dataset_kwargs)
    test_dataset = MultimodalSpatialDataset(test_rows, **dataset_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    metadata_dim = len(METADATA_COLUMNS) + len(class_to_id)
    model = MultimodalSpatialRegressor(
        architecture=architecture,
        metadata_dim=metadata_dim,
        output_dim=len(TARGET_COLUMNS),
        use_pretrained=use_pretrained,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    history = []
    best_val_position_error = math.inf
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "multimodal_spatial_best.pt"

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            targets = batch["target"].to(device)
            optimizer.zero_grad()
            predictions = model(**_model_inputs(batch, device))
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
            payload = _checkpoint_payload(
                model=model,
                architecture=architecture,
                class_to_id=class_to_id,
                target_normalizer=target_normalizer,
                metadata_normalizer=metadata_normalizer,
                image_size=image_size,
                context_image_size=context_image_size,
                minimum_depth_m=minimum_depth_m,
                maximum_depth_m=maximum_depth_m,
                use_pretrained=use_pretrained,
            )
            payload["val_metrics"] = val_metrics
            torch.save(payload, best_model_path)

        print(
            f"architecture={architecture} epoch={epoch + 1}/{epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_distance={val_metrics['mae_distance_camera_m']:.4f} "
            f"val_pos_err={val_metrics['mean_position_euclidean_error_m']:.4f} "
            f"val_dim_err={val_metrics['mean_dimension_euclidean_error_m']:.4f}"
        )

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = _evaluate(model, test_loader, device, target_normalizer)

    final_model_path = output_dir / "multimodal_spatial_final.pt"
    final_payload = _checkpoint_payload(
        model=model,
        architecture=architecture,
        class_to_id=class_to_id,
        target_normalizer=target_normalizer,
        metadata_normalizer=metadata_normalizer,
        image_size=image_size,
        context_image_size=context_image_size,
        minimum_depth_m=minimum_depth_m,
        maximum_depth_m=maximum_depth_m,
        use_pretrained=use_pretrained,
    )
    final_payload["test_metrics"] = test_metrics
    torch.save(final_payload, final_model_path)

    summary = {
        "model_family": MODEL_FAMILY,
        "model_version": MODEL_VERSION,
        "architecture": architecture,
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
        "target_normalizer": target_normalizer.to_dict(),
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "use_pretrained": use_pretrained,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "context_image_size": context_image_size,
        "minimum_depth_m": minimum_depth_m,
        "maximum_depth_m": maximum_depth_m,
        "best_val_position_error_m": best_val_position_error,
        "test_metrics": test_metrics,
        "history": history,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    summary["summary_path"] = str(summary_path)
    return summary


def _load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if checkpoint.get("model_family") != MODEL_FAMILY:
        raise ValueError(
            f"Checkpoint is not a {MODEL_FAMILY} model: {checkpoint_path}"
        )

    target_normalizer = _normalizer_from_dict(
        TargetNormalizer,
        checkpoint["target_normalizer"],
    )
    metadata_normalizer = _normalizer_from_dict(
        MetadataNormalizer,
        checkpoint["metadata_normalizer"],
    )
    metadata_dim = len(checkpoint["metadata_columns"]) + len(checkpoint["class_to_id"])
    model = MultimodalSpatialRegressor(
        architecture=checkpoint["architecture"],
        metadata_dim=metadata_dim,
        output_dim=len(checkpoint["target_columns"]),
        use_pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, target_normalizer, metadata_normalizer


def _write_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No prediction rows to write.")

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def export_multimodal_spatial_predictions(
    crop_labels_csv: str | Path,
    checkpoint_path: str | Path,
    output_csv: str | Path,
    batch_size: int = 16,
) -> Path:
    rows = _read_rows(crop_labels_csv)

    if not rows:
        raise ValueError("No crop rows found.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, checkpoint, target_normalizer, metadata_normalizer = _load_checkpoint(
        checkpoint_path,
        device,
    )
    architecture = checkpoint["architecture"]
    class_to_id = checkpoint["class_to_id"]
    _validate_split(rows, class_to_id, "Prediction split", architecture)

    dataset = MultimodalSpatialDataset(
        rows=rows,
        architecture=architecture,
        class_to_id=class_to_id,
        target_normalizer=target_normalizer,
        metadata_normalizer=metadata_normalizer,
        image_size=int(checkpoint["image_size"]),
        context_image_size=int(checkpoint["context_image_size"]),
        minimum_depth_m=float(checkpoint["minimum_depth_m"]),
        maximum_depth_m=float(checkpoint["maximum_depth_m"]),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    output_rows: list[dict[str, Any] | None] = [None] * len(rows)

    with torch.no_grad():
        for batch in dataloader:
            predictions_normalized = model(**_model_inputs(batch, device))
            predictions = target_normalizer.denormalize(predictions_normalized).cpu()
            indices = batch["index"].tolist()

            for batch_index, row_index in enumerate(indices):
                row = rows[row_index]
                output_row: dict[str, Any] = {
                    "architecture": architecture,
                    "crop_image_path": row.get("crop_image_path", ""),
                    "source_image_path": row.get("source_image_path", ""),
                    "depth_crop_path": row.get("depth_crop_path", ""),
                    "frame_id": row["frame_id"],
                    "scene_id": row["scene_id"],
                    "class_name": row["class_name"],
                    "semantic_class": row["semantic_class"],
                    "is_focused_object": row["is_focused_object"],
                }

                for target_index, column in enumerate(checkpoint["target_columns"]):
                    true_value = float(row[column])
                    predicted_value = float(predictions[batch_index, target_index])
                    output_row[f"true_{column}"] = true_value
                    output_row[f"pred_{column}"] = predicted_value
                    output_row[f"abs_error_{column}"] = abs(
                        true_value - predicted_value
                    )

                output_rows[row_index] = output_row

    return _write_csv(
        [row for row in output_rows if row is not None],
        output_csv,
    )
