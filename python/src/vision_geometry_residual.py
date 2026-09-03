from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json
import math
import random
from typing import Any

from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .vision_crop_spatial_baseline import (
    CLASS_COLUMN,
    CLASS_ENCODING,
    MetadataNormalizer,
    TargetNormalizer,
    build_metadata_tensor,
)
from .vision_multimodal_spatial import (
    DepthEncoder,
    _load_depth_tensor,
    _resnet18_encoder,
    _rgb_crop_transform,
)


GEOMETRY_MLP = "geometry_mlp"
GEOMETRY_RGBD = "geometry_rgbd"
GEOMETRY_RESIDUAL_ARCHITECTURES = (GEOMETRY_MLP, GEOMETRY_RGBD)

MODEL_FAMILY = "geometry_residual"
MODEL_VERSION = 1

GEOMETRY_FEATURE_COLUMNS = [
    "raw_position_camera_x",
    "raw_position_camera_y",
    "raw_position_camera_z",
    "raw_dimension_x",
    "raw_dimension_y",
    "raw_dimension_z",
    "robust_position_camera_x",
    "robust_position_camera_y",
    "robust_position_camera_z",
    "robust_dimension_x",
    "robust_dimension_y",
    "robust_dimension_z",
    "surface_depth_min_m",
    "surface_depth_max_m",
    "surface_depth_p_lower_m",
    "surface_depth_p_upper_m",
    "target_mask_fraction",
    "target_depth_valid_fraction",
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_center_x_norm",
    "bbox_center_y_norm",
    "bbox_area_normalized",
    "bbox_aspect_ratio",
]

RESIDUAL_COLUMNS = [
    "position_camera_x",
    "position_camera_y",
    "position_camera_z",
    "dimension_x",
    "dimension_y",
    "dimension_z",
]


def _validate_architecture(architecture: str) -> None:
    if architecture not in GEOMETRY_RESIDUAL_ARCHITECTURES:
        raise ValueError(
            f"Unsupported geometry residual architecture '{architecture}'. "
            f"Choose from: {', '.join(GEOMETRY_RESIDUAL_ARCHITECTURES)}"
        )


def _uses_images(architecture: str) -> bool:
    return architecture == GEOMETRY_RGBD


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _class_mapping(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        class_name: index
        for index, class_name in enumerate(
            sorted({str(row[CLASS_COLUMN]) for row in rows})
        )
    }


def _validate_rows(
    rows: list[dict[str, Any]],
    class_to_id: dict[str, int],
    architecture: str,
    split_name: str,
) -> None:
    unknown_classes = sorted(
        {str(row.get(CLASS_COLUMN, "")) for row in rows} - set(class_to_id)
    )
    if unknown_classes:
        raise ValueError(
            f"{split_name} contains classes absent from training: {unknown_classes}"
        )

    required_values = [
        *GEOMETRY_FEATURE_COLUMNS,
        *[f"true_{column}" for column in RESIDUAL_COLUMNS],
    ]

    for row_index, row in enumerate(rows):
        for column in required_values:
            if str(row.get(column, "")).strip() == "":
                raise ValueError(
                    f"{split_name} row {row_index} is missing required {column}."
                )

        if _uses_images(architecture):
            for column in ("crop_image_path", "masked_depth_crop_path"):
                value = str(row.get(column, "")).strip()
                if not value:
                    raise ValueError(
                        f"{split_name} row {row_index} is missing required {column}."
                    )
                if not Path(value).expanduser().exists():
                    raise FileNotFoundError(
                        f"{split_name} row {row_index} {column} does not exist: {value}"
                    )


def _raw_geometry(row: dict[str, Any]) -> list[float]:
    return [
        float(row[f"raw_{column}"])
        for column in RESIDUAL_COLUMNS
    ]


def _true_geometry(row: dict[str, Any]) -> list[float]:
    return [
        float(row[f"true_{column}"])
        for column in RESIDUAL_COLUMNS
    ]


def _residual_targets(rows: list[dict[str, Any]]) -> list[list[float]]:
    return [
        [true - raw for true, raw in zip(_true_geometry(row), _raw_geometry(row))]
        for row in rows
    ]


class GeometryResidualDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        architecture: str,
        class_to_id: dict[str, int],
        feature_normalizer: MetadataNormalizer,
        residual_normalizer: TargetNormalizer,
        image_size: int,
        minimum_depth_m: float,
        maximum_depth_m: float,
    ):
        _validate_architecture(architecture)
        self.rows = rows
        self.architecture = architecture
        self.class_to_id = class_to_id
        self.feature_normalizer = feature_normalizer
        self.residual_normalizer = residual_normalizer
        self.image_size = image_size
        self.minimum_depth_m = minimum_depth_m
        self.maximum_depth_m = maximum_depth_m
        self.rgb_transform = _rgb_crop_transform(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        raw_geometry = torch.tensor(_raw_geometry(row), dtype=torch.float32)
        true_geometry = torch.tensor(_true_geometry(row), dtype=torch.float32)
        residual = true_geometry - raw_geometry

        sample: dict[str, torch.Tensor] = {
            "index": torch.tensor(index, dtype=torch.long),
            "features": build_metadata_tensor(
                row=row,
                class_to_id=self.class_to_id,
                metadata_columns=GEOMETRY_FEATURE_COLUMNS,
                class_encoding=CLASS_ENCODING,
                metadata_normalizer=self.feature_normalizer,
            ),
            "raw_geometry": raw_geometry,
            "true_geometry": true_geometry,
            "true_distance": torch.tensor(
                float(row["true_distance_camera_m"]),
                dtype=torch.float32,
            ),
            "target_residual": self.residual_normalizer.normalize(residual),
        }

        if _uses_images(self.architecture):
            with Image.open(row["crop_image_path"]) as image:
                sample["rgb_crop"] = self.rgb_transform(image.convert("RGB"))
            sample["masked_depth"] = _load_depth_tensor(
                path=row["masked_depth_crop_path"],
                image_size=self.image_size,
                minimum_depth_m=self.minimum_depth_m,
                maximum_depth_m=self.maximum_depth_m,
            )

        return sample


class GeometryResidualRegressor(nn.Module):
    def __init__(
        self,
        architecture: str,
        feature_dim: int,
        output_dim: int = len(RESIDUAL_COLUMNS),
        use_pretrained: bool = True,
    ):
        super().__init__()
        _validate_architecture(architecture)
        self.architecture = architecture
        fused_dim = feature_dim
        self.rgb_encoder: nn.Module | None = None
        self.depth_encoder: DepthEncoder | None = None

        if _uses_images(architecture):
            self.rgb_encoder, rgb_dim = _resnet18_encoder(use_pretrained)
            self.depth_encoder = DepthEncoder(output_dim=128)
            fused_dim += rgb_dim + self.depth_encoder.output_dim

        hidden_dim = 256 if _uses_images(architecture) else 128
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, output_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        rgb_crop: torch.Tensor | None = None,
        masked_depth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused = [features]

        if self.rgb_encoder is not None:
            if rgb_crop is None or masked_depth is None or self.depth_encoder is None:
                raise ValueError("RGB and masked depth are required by geometry_rgbd.")
            fused.append(self.rgb_encoder(rgb_crop))
            fused.append(self.depth_encoder(masked_depth))

        return self.head(torch.cat(fused, dim=1))


def _model_inputs(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    return {
        "features": batch["features"].to(device),
        "rgb_crop": batch["rgb_crop"].to(device) if "rgb_crop" in batch else None,
        "masked_depth": batch["masked_depth"].to(device) if "masked_depth" in batch else None,
    }


def _corrected_geometry(
    model: GeometryResidualRegressor,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    residual_normalizer: TargetNormalizer,
) -> torch.Tensor:
    residual_normalized = model(**_model_inputs(batch, device))
    residual = residual_normalizer.denormalize(residual_normalized)
    raw_geometry = batch["raw_geometry"].to(device)
    corrected = raw_geometry + residual
    corrected[:, 3:6] = torch.clamp(corrected[:, 3:6], min=1e-4)
    return corrected


def _metrics_from_tensors(
    predictions: torch.Tensor,
    truths: torch.Tensor,
    true_distances: torch.Tensor,
) -> dict[str, float]:
    absolute_errors = torch.abs(predictions - truths)
    predicted_distances = torch.linalg.vector_norm(predictions[:, 0:3], dim=1)
    distance_errors = torch.abs(predicted_distances - true_distances)
    position_errors = torch.linalg.vector_norm(
        predictions[:, 0:3] - truths[:, 0:3],
        dim=1,
    )
    dimension_errors = torch.linalg.vector_norm(
        predictions[:, 3:6] - truths[:, 3:6],
        dim=1,
    )

    metrics = {
        "mae_distance_camera_m": float(distance_errors.mean()),
        "mean_position_euclidean_error_m": float(position_errors.mean()),
        "mean_dimension_euclidean_error_m": float(dimension_errors.mean()),
    }
    for index, axis in enumerate(("x", "y", "z")):
        metrics[f"mae_position_camera_{axis}"] = float(
            absolute_errors[:, index].mean()
        )
        metrics[f"mae_dimension_{axis}"] = float(
            absolute_errors[:, index + 3].mean()
        )
    return metrics


def _evaluate(
    model: GeometryResidualRegressor,
    dataloader: DataLoader,
    device: torch.device,
    residual_normalizer: TargetNormalizer,
) -> dict[str, float]:
    model.eval()
    predictions = []
    truths = []
    distances = []

    with torch.no_grad():
        for batch in dataloader:
            predictions.append(
                _corrected_geometry(model, batch, device, residual_normalizer).cpu()
            )
            truths.append(batch["true_geometry"])
            distances.append(batch["true_distance"])

    return _metrics_from_tensors(
        torch.cat(predictions),
        torch.cat(truths),
        torch.cat(distances),
    )


def _normalizer_from_dict(normalizer_type, data: dict[str, list[float]]):
    return normalizer_type(
        mean=torch.tensor(data["mean"], dtype=torch.float32),
        std=torch.tensor(data["std"], dtype=torch.float32),
    )


def _checkpoint_payload(
    model: GeometryResidualRegressor,
    architecture: str,
    class_to_id: dict[str, int],
    feature_normalizer: MetadataNormalizer,
    residual_normalizer: TargetNormalizer,
    image_size: int,
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
        "feature_columns": GEOMETRY_FEATURE_COLUMNS,
        "residual_columns": RESIDUAL_COLUMNS,
        "feature_normalizer": feature_normalizer.to_dict(),
        "residual_normalizer": residual_normalizer.to_dict(),
        "image_size": image_size,
        "minimum_depth_m": minimum_depth_m,
        "maximum_depth_m": maximum_depth_m,
        "use_pretrained": use_pretrained,
    }


def train_geometry_residual_from_splits(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    output_dir: str | Path,
    architecture: str,
    epochs: int = 50,
    batch_size: int = 16,
    image_size: int = 128,
    minimum_depth_m: float = 0.3,
    maximum_depth_m: float = 3.0,
    seed: int = 42,
    use_pretrained: bool = True,
    early_stopping_patience: int = 12,
) -> dict[str, Any]:
    _validate_architecture(architecture)
    train_rows = _read_rows(train_csv)
    val_rows = _read_rows(val_csv)
    test_rows = _read_rows(test_csv)

    if len(train_rows) < 10 or not val_rows or not test_rows:
        raise ValueError("Geometry residual training requires non-empty explicit splits.")

    random.seed(seed)
    torch.manual_seed(seed)
    class_to_id = _class_mapping(train_rows)
    _validate_rows(train_rows, class_to_id, architecture, "Training split")
    _validate_rows(val_rows, class_to_id, architecture, "Validation split")
    _validate_rows(test_rows, class_to_id, architecture, "Test split")

    feature_normalizer = MetadataNormalizer.from_rows(
        train_rows,
        GEOMETRY_FEATURE_COLUMNS,
    )
    residual_normalizer = TargetNormalizer.from_targets(
        _residual_targets(train_rows)
    )
    dataset_kwargs = {
        "architecture": architecture,
        "class_to_id": class_to_id,
        "feature_normalizer": feature_normalizer,
        "residual_normalizer": residual_normalizer,
        "image_size": image_size,
        "minimum_depth_m": minimum_depth_m,
        "maximum_depth_m": maximum_depth_m,
    }
    train_dataset = GeometryResidualDataset(train_rows, **dataset_kwargs)
    val_dataset = GeometryResidualDataset(val_rows, **dataset_kwargs)
    test_dataset = GeometryResidualDataset(test_rows, **dataset_kwargs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    feature_dim = len(GEOMETRY_FEATURE_COLUMNS) + len(class_to_id)
    model = GeometryResidualRegressor(
        architecture=architecture,
        feature_dim=feature_dim,
        use_pretrained=use_pretrained,
    ).to(device)
    learning_rate = 1e-4 if _uses_images(architecture) else 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "geometry_residual_best.pt"
    history = []
    best_val_score = math.inf
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            predictions = model(**_model_inputs(batch, device))
            targets = batch["target_residual"].to(device)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        train_loss = total_loss / max(1, len(train_loader))
        val_metrics = _evaluate(model, val_loader, device, residual_normalizer)
        val_score = (
            val_metrics["mean_position_euclidean_error_m"]
            + val_metrics["mean_dimension_euclidean_error_m"]
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "selection_score_m": val_score,
            "val_metrics": val_metrics,
        })

        if val_score < best_val_score:
            best_val_score = val_score
            epochs_without_improvement = 0
            payload = _checkpoint_payload(
                model=model,
                architecture=architecture,
                class_to_id=class_to_id,
                feature_normalizer=feature_normalizer,
                residual_normalizer=residual_normalizer,
                image_size=image_size,
                minimum_depth_m=minimum_depth_m,
                maximum_depth_m=maximum_depth_m,
                use_pretrained=use_pretrained,
            )
            payload["val_metrics"] = val_metrics
            payload["selection_score_m"] = val_score
            torch.save(payload, best_model_path)
        else:
            epochs_without_improvement += 1

        print(
            f"architecture={architecture} epoch={epoch + 1}/{epochs} "
            f"loss={train_loss:.5f} "
            f"val_distance={val_metrics['mae_distance_camera_m']:.4f} "
            f"val_position={val_metrics['mean_position_euclidean_error_m']:.4f} "
            f"val_dimensions={val_metrics['mean_dimension_euclidean_error_m']:.4f} "
            f"score={val_score:.4f}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = _evaluate(model, test_loader, device, residual_normalizer)

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
        "feature_columns": GEOMETRY_FEATURE_COLUMNS,
        "residual_columns": RESIDUAL_COLUMNS,
        "best_model_path": str(best_model_path),
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "batch_size": batch_size,
        "image_size": image_size,
        "minimum_depth_m": minimum_depth_m,
        "maximum_depth_m": maximum_depth_m,
        "selection_metric": "position_euclidean_error_m + dimension_euclidean_error_m",
        "best_val_selection_score_m": best_val_score,
        "best_val_metrics": checkpoint["val_metrics"],
        "test_metrics": test_metrics,
        "history": history,
    }

    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)

    return summary


def _load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[
    GeometryResidualRegressor,
    dict[str, Any],
    MetadataNormalizer,
    TargetNormalizer,
]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("model_family") != MODEL_FAMILY:
        raise ValueError(f"Checkpoint is not a {MODEL_FAMILY} model: {checkpoint_path}")

    feature_normalizer = _normalizer_from_dict(
        MetadataNormalizer,
        checkpoint["feature_normalizer"],
    )
    residual_normalizer = _normalizer_from_dict(
        TargetNormalizer,
        checkpoint["residual_normalizer"],
    )
    model = GeometryResidualRegressor(
        architecture=checkpoint["architecture"],
        feature_dim=len(checkpoint["feature_columns"]) + len(checkpoint["class_to_id"]),
        use_pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, feature_normalizer, residual_normalizer


def _prediction_rows_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    predictions = torch.tensor([
        [float(row[f"pred_{column}"]) for column in RESIDUAL_COLUMNS]
        for row in rows
    ])
    truths = torch.tensor([
        [float(row[f"true_{column}"]) for column in RESIDUAL_COLUMNS]
        for row in rows
    ])
    distances = torch.tensor([
        float(row["true_distance_camera_m"])
        for row in rows
    ])
    return _metrics_from_tensors(predictions, truths, distances)


def export_geometry_residual_predictions(
    geometry_csv: str | Path,
    checkpoint_path: str | Path,
    predictions_output: str | Path,
    summary_output: str | Path,
    batch_size: int = 16,
) -> dict[str, Any]:
    rows = _read_rows(geometry_csv)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, checkpoint, feature_normalizer, residual_normalizer = _load_checkpoint(
        checkpoint_path,
        device,
    )
    architecture = checkpoint["architecture"]
    _validate_rows(rows, checkpoint["class_to_id"], architecture, "Prediction split")
    dataset = GeometryResidualDataset(
        rows=rows,
        architecture=architecture,
        class_to_id=checkpoint["class_to_id"],
        feature_normalizer=feature_normalizer,
        residual_normalizer=residual_normalizer,
        image_size=checkpoint["image_size"],
        minimum_depth_m=checkpoint["minimum_depth_m"],
        maximum_depth_m=checkpoint["maximum_depth_m"],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    output_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            predictions = _corrected_geometry(
                model,
                batch,
                device,
                residual_normalizer,
            ).cpu()
            indices = batch["index"].tolist()

            for batch_index, row_index in enumerate(indices):
                source = rows[row_index]
                predicted = predictions[batch_index]
                truth = batch["true_geometry"][batch_index]
                predicted_distance = float(torch.linalg.vector_norm(predicted[0:3]))
                true_distance = float(batch["true_distance"][batch_index])
                output: dict[str, Any] = {
                    "architecture": architecture,
                    "frame_id": source.get("frame_id", ""),
                    "scene_id": source.get("scene_id", ""),
                    "object_name": source.get("object_name", ""),
                    "semantic_class": source.get("semantic_class", ""),
                    "crop_image_path": source.get("crop_image_path", ""),
                    "masked_depth_crop_path": source.get("masked_depth_crop_path", ""),
                    "true_distance_camera_m": true_distance,
                    "pred_distance_camera_m": predicted_distance,
                    "abs_error_distance_camera_m": abs(predicted_distance - true_distance),
                }

                for index, column in enumerate(RESIDUAL_COLUMNS):
                    true_value = float(truth[index])
                    predicted_value = float(predicted[index])
                    output[f"true_{column}"] = true_value
                    output[f"pred_{column}"] = predicted_value
                    output[f"abs_error_{column}"] = abs(predicted_value - true_value)

                output_rows.append(output)

    predictions_output = Path(predictions_output).expanduser().resolve()
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    with predictions_output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    overall_metrics = _prediction_rows_metrics(output_rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output_rows:
        grouped[str(row["semantic_class"])].append(row)

    per_class = {}
    for class_name, class_rows in sorted(grouped.items()):
        per_class[class_name] = {
            "num_rows": len(class_rows),
            **_prediction_rows_metrics(class_rows),
        }

    distance_bin_definitions = (
        ("near_lt_1m", 0.0, 1.0),
        ("middle_1m_to_1_5m", 1.0, 1.5),
        ("far_ge_1_5m", 1.5, math.inf),
    )
    distance_bins = {}
    for label, lower, upper in distance_bin_definitions:
        bin_rows = [
            row
            for row in output_rows
            if lower <= float(row["true_distance_camera_m"]) < upper
        ]
        if bin_rows:
            distance_bins[label] = {
                "num_rows": len(bin_rows),
                **_prediction_rows_metrics(bin_rows),
            }

    size_bin_definitions = (
        ("small_diagonal_lt_0_3m", 0.0, 0.3),
        ("medium_diagonal_0_3m_to_0_45m", 0.3, 0.45),
        ("large_diagonal_ge_0_45m", 0.45, math.inf),
    )
    size_bins = {}
    for label, lower, upper in size_bin_definitions:
        bin_rows = []
        for row in output_rows:
            diagonal = math.sqrt(sum(
                float(row[f"true_dimension_{axis}"]) ** 2
                for axis in ("x", "y", "z")
            ))
            if lower <= diagonal < upper:
                bin_rows.append(row)
        if bin_rows:
            size_bins[label] = {
                "num_rows": len(bin_rows),
                **_prediction_rows_metrics(bin_rows),
            }

    summary = {
        "model_family": MODEL_FAMILY,
        "architecture": architecture,
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "geometry_csv": str(Path(geometry_csv).expanduser().resolve()),
        "predictions_path": str(predictions_output),
        "num_rows": len(output_rows),
        "test_metrics": overall_metrics,
        "per_class": per_class,
        "distance_bins": distance_bins,
        "object_size_bins": size_bins,
    }
    summary_output = Path(summary_output).expanduser().resolve()
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w") as file:
        json.dump(summary, file, indent=2)

    return summary
