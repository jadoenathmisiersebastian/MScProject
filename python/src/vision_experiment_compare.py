from __future__ import annotations

from pathlib import Path
import json


def load_json(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()

    with path.open("r") as f:
        return json.load(f)


def _evaluation_metrics(report: dict) -> dict:
    """Return final-test metrics from both current and legacy report formats."""
    test_metrics = report.get("test_metrics")
    if isinstance(test_metrics, dict):
        return test_metrics

    legacy_metrics = report.get("metrics")
    if isinstance(legacy_metrics, dict):
        return legacy_metrics

    return report


def compare_vision_experiments(
    bbox_report_path: str | Path,
    crop_summary_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    bbox_report_path = Path(bbox_report_path).expanduser().resolve()
    crop_summary_path = Path(crop_summary_path).expanduser().resolve()

    bbox_report = load_json(bbox_report_path)
    crop_summary = load_json(crop_summary_path)

    distance_metrics = _evaluation_metrics(bbox_report["distance_baseline"])
    position_metrics = _evaluation_metrics(bbox_report["position_baseline"])
    dimensions_metrics = _evaluation_metrics(bbox_report["dimensions_baseline"])
    crop_metrics = _evaluation_metrics(crop_summary)
    candidate_architecture = crop_summary.get("architecture", "rgb_crop_v2")

    candidate_metrics = {
        "distance_mae_m": crop_metrics["mae_distance_camera_m"],
        "position_mean_euclidean_error_m": crop_metrics["mean_position_euclidean_error_m"],
        "dimensions_mean_euclidean_error_m": crop_metrics.get("mean_dimension_euclidean_error_m"),
        "position_mae_per_axis_m": {
            "position_camera_x": crop_metrics["mae_position_camera_x"],
            "position_camera_y": crop_metrics["mae_position_camera_y"],
            "position_camera_z": crop_metrics["mae_position_camera_z"],
        },
        "dimensions_mae_per_axis_m": {
            "dimension_x": crop_metrics["mae_dimension_x"],
            "dimension_y": crop_metrics["mae_dimension_y"],
            "dimension_z": crop_metrics["mae_dimension_z"],
        },
    }

    comparison = {
        "bbox_report_path": str(bbox_report_path),
        "crop_summary_path": str(crop_summary_path),
        "candidate_architecture": candidate_architecture,
        "bbox_feature_baseline": {
            "distance_mae_m": distance_metrics["mae_m"],
            "position_mean_euclidean_error_m": position_metrics["mean_euclidean_error_m"],
            "dimensions_mean_euclidean_error_m": dimensions_metrics["mean_euclidean_error_m"],
            "position_mae_per_axis_m": position_metrics["mae_per_axis_m"],
            "dimensions_mae_per_axis_m": dimensions_metrics["mae_per_axis_m"],
        },
        "candidate_model": candidate_metrics,
    }

    bbox_distance = comparison["bbox_feature_baseline"]["distance_mae_m"]
    crop_distance = comparison["candidate_model"]["distance_mae_m"]

    comparison["distance_mae_delta_m"] = crop_distance - bbox_distance
    comparison["distance_mae_improved_by_crop"] = crop_distance < bbox_distance
    comparison["candidate_distance_mae_delta_m"] = crop_distance - bbox_distance
    comparison["candidate_improved_distance_mae"] = crop_distance < bbox_distance

    if output_path is None:
        output_path = crop_summary_path.with_name("experiment_comparison.json")
    else:
        output_path = Path(output_path).expanduser().resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(comparison, f, indent=2)

    return output_path
