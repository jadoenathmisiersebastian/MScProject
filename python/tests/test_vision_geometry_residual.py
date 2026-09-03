from __future__ import annotations

from pathlib import Path
import csv
import tempfile
import unittest

import torch

from src.vision_geometry_residual import (
    GEOMETRY_FEATURE_COLUMNS,
    GEOMETRY_MLP,
    GEOMETRY_RGBD,
    GeometryResidualRegressor,
    export_geometry_residual_predictions,
    train_geometry_residual_from_splits,
)


def _row(index: int, class_name: str) -> dict[str, object]:
    raw_position = [0.002 * index, -0.05, 1.0 + 0.01 * index]
    raw_dimensions = [0.10, 0.18, 0.11]
    class_offset = 0.01 if class_name == "bottle" else 0.02
    true_position = [
        raw_position[0] + 0.001,
        raw_position[1] - 0.002,
        raw_position[2] + class_offset,
    ]
    true_dimensions = [
        raw_dimensions[0] + class_offset,
        raw_dimensions[1] + 0.005,
        raw_dimensions[2] + class_offset,
    ]
    row: dict[str, object] = {
        "frame_id": f"frame_{index:06d}",
        "scene_id": "test",
        "object_name": class_name,
        "semantic_class": class_name,
        "true_distance_camera_m": sum(value * value for value in true_position) ** 0.5,
    }

    for axis, value in zip(("x", "y", "z"), raw_position):
        row[f"raw_position_camera_{axis}"] = value
        row[f"robust_position_camera_{axis}"] = value + 0.002
    for axis, value in zip(("x", "y", "z"), raw_dimensions):
        row[f"raw_dimension_{axis}"] = value
        row[f"robust_dimension_{axis}"] = value - 0.002
    for axis, value in zip(("x", "y", "z"), true_position):
        row[f"true_position_camera_{axis}"] = value
    for axis, value in zip(("x", "y", "z"), true_dimensions):
        row[f"true_dimension_{axis}"] = value

    row.update({
        "surface_depth_min_m": raw_position[2] - 0.05,
        "surface_depth_max_m": raw_position[2] + 0.05,
        "surface_depth_p_lower_m": raw_position[2] - 0.04,
        "surface_depth_p_upper_m": raw_position[2] + 0.04,
        "target_mask_fraction": 0.4,
        "target_depth_valid_fraction": 1.0,
        "bbox_width_norm": 0.1,
        "bbox_height_norm": 0.2,
        "bbox_center_x_norm": 0.5,
        "bbox_center_y_norm": 0.55,
        "bbox_area_normalized": 0.02,
        "bbox_aspect_ratio": 0.5,
    })
    return row


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class VisionGeometryResidualTests(unittest.TestCase):
    def test_geometry_residual_models_produce_six_corrections(self):
        feature_dim = len(GEOMETRY_FEATURE_COLUMNS) + 2
        features = torch.zeros((2, feature_dim))

        mlp = GeometryResidualRegressor(
            architecture=GEOMETRY_MLP,
            feature_dim=feature_dim,
            use_pretrained=False,
        )
        self.assertEqual(tuple(mlp(features).shape), (2, 6))

        rgbd = GeometryResidualRegressor(
            architecture=GEOMETRY_RGBD,
            feature_dim=feature_dim,
            use_pretrained=False,
        )
        rgbd.eval()
        with torch.no_grad():
            output = rgbd(
                features,
                rgb_crop=torch.zeros((2, 3, 64, 64)),
                masked_depth=torch.zeros((2, 2, 64, 64)),
            )
        self.assertEqual(tuple(output.shape), (2, 6))

    def test_geometry_mlp_trains_and_exports_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.csv"
            val_path = root / "val.csv"
            test_path = root / "test.csv"
            train_rows = [
                _row(index, "bottle" if index % 2 == 0 else "glass")
                for index in range(12)
            ]
            val_rows = [
                _row(index + 20, "bottle" if index % 2 == 0 else "glass")
                for index in range(4)
            ]
            test_rows = [
                _row(index + 30, "bottle" if index % 2 == 0 else "glass")
                for index in range(4)
            ]
            _write_rows(train_path, train_rows)
            _write_rows(val_path, val_rows)
            _write_rows(test_path, test_rows)

            model_dir = root / "model"
            summary = train_geometry_residual_from_splits(
                train_csv=train_path,
                val_csv=val_path,
                test_csv=test_path,
                output_dir=model_dir,
                architecture=GEOMETRY_MLP,
                epochs=2,
                batch_size=4,
                use_pretrained=False,
            )

            checkpoint = model_dir / "geometry_residual_best.pt"
            self.assertTrue(checkpoint.exists())
            self.assertEqual(summary["num_train"], 12)
            self.assertEqual(summary["num_test"], 4)

            predictions = root / "predictions.csv"
            prediction_summary = root / "prediction_summary.json"
            exported = export_geometry_residual_predictions(
                geometry_csv=test_path,
                checkpoint_path=checkpoint,
                predictions_output=predictions,
                summary_output=prediction_summary,
                batch_size=2,
            )

            self.assertEqual(exported["num_rows"], 4)
            self.assertTrue(predictions.exists())
            self.assertTrue(prediction_summary.exists())


if __name__ == "__main__":
    unittest.main()
