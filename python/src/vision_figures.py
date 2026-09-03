from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POSITION_AXES = ("x", "y", "z")
DIMENSION_AXES = ("x", "y", "z")


class FigureGenerationError(RuntimeError):
    pass


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def _finite_pairs(rows: Iterable[dict[str, str]], true_key: str, pred_key: str) -> tuple[list[float], list[float]]:
    true_values: list[float] = []
    pred_values: list[float] = []

    for row in rows:
        true_value = _float(row, true_key)
        pred_value = _float(row, pred_key)
        if math.isfinite(true_value) and math.isfinite(pred_value):
            true_values.append(true_value)
            pred_values.append(pred_value)

    return true_values, pred_values


def _finite_values(rows: Iterable[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _float(row, key)
        if math.isfinite(value):
            values.append(value)
    return values


def _save_current(path: Path, rect: tuple[float, float, float, float] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rect is None:
        plt.tight_layout()
    else:
        plt.tight_layout(rect=rect)
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _identity_limits(a: list[float], b: list[float]) -> tuple[float, float]:
    values = a + b
    if not values:
        return 0.0, 1.0

    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        padding = 0.05 if math.isclose(low, 0.0) else abs(low) * 0.05
    else:
        padding = (high - low) * 0.08

    return low - padding, high + padding


def plot_true_vs_pred_distance(rows: list[dict[str, str]], output_dir: Path) -> Path:
    true_values, pred_values = _finite_pairs(rows, "true_distance_camera_m", "pred_distance_camera_m")
    if not true_values:
        raise FigureGenerationError("No distance prediction columns found.")

    low, high = _identity_limits(true_values, pred_values)

    plt.figure(figsize=(6.4, 5.2))
    plt.scatter(true_values, pred_values, s=18, alpha=0.65, edgecolors="none")
    plt.plot([low, high], [low, high], color="black", linewidth=1.2, linestyle="--", label="Perfect Prediction")
    plt.xlabel("True Camera Distance (m)")
    plt.ylabel("Predicted Camera Distance (m)")
    plt.title("Camera Distance Prediction")
    plt.xlim(low, high)
    plt.ylim(low, high)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    return _save_current(output_dir / "true_vs_pred_distance.png")


def plot_distance_error_histogram(rows: list[dict[str, str]], output_dir: Path) -> Path:
    errors = _finite_values(rows, "abs_error_distance_camera_m")
    if not errors:
        true_values, pred_values = _finite_pairs(rows, "true_distance_camera_m", "pred_distance_camera_m")
        errors = [abs(t - p) for t, p in zip(true_values, pred_values)]

    if not errors:
        raise FigureGenerationError("No distance error columns found.")

    plt.figure(figsize=(6.4, 4.6))
    plt.hist(errors, bins=24, color="#4c78a8", alpha=0.86)
    plt.axvline(sum(errors) / len(errors), color="black", linestyle="--", linewidth=1.2, label="Mean Error")
    plt.xlabel("Absolute Distance Error (m)")
    plt.ylabel("Object Crops")
    plt.title("Distance Error Distribution")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(frameon=False)
    return _save_current(output_dir / "distance_error_histogram.png")


def plot_position_error_boxplot(rows: list[dict[str, str]], output_dir: Path) -> Path:
    axis_errors: list[list[float]] = []

    for axis in POSITION_AXES:
        error_key = f"abs_error_position_camera_{axis}"
        errors = _finite_values(rows, error_key)

        if not errors:
            true_values, pred_values = _finite_pairs(
                rows,
                f"true_position_camera_{axis}",
                f"pred_position_camera_{axis}",
            )
            errors = [abs(t - p) for t, p in zip(true_values, pred_values)]

        axis_errors.append(errors)

    if not any(axis_errors):
        raise FigureGenerationError("No camera-relative position error columns found.")

    plt.figure(figsize=(6.4, 4.8))
    box = plt.boxplot(
        axis_errors,
        tick_labels=["X", "Y", "Z"],
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=False,
        widths=0.55,
    )

    colors = ["#4e79a7", "#59a14f", "#e15759"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.3)

    for mean in box["means"]:
        mean.set_color("black")
        mean.set_linewidth(1.1)
        mean.set_linestyle("--")

    plt.xlabel("Camera-Relative Position Axis")
    plt.ylabel("Absolute Position Error (m)")
    plt.title("Camera-Relative Position Error by Axis")
    plt.grid(True, axis="y", alpha=0.25)
    return _save_current(output_dir / "position_error_by_axis.png")


def plot_dimension_scatter(rows: list[dict[str, str]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharex=False, sharey=False)

    plotted = False
    for axis, dim in zip(axes, DIMENSION_AXES):
        true_values, pred_values = _finite_pairs(rows, f"true_dimension_{dim}", f"pred_dimension_{dim}")
        if not true_values:
            axis.set_visible(False)
            continue

        plotted = True
        low, high = _identity_limits(true_values, pred_values)
        axis.scatter(true_values, pred_values, s=14, alpha=0.62, edgecolors="none")
        axis.plot([low, high], [low, high], color="black", linewidth=1.0, linestyle="--")
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_title(f"Dimension {dim.upper()}")
        axis.set_xlabel("True (m)")
        axis.set_ylabel("Predicted (m)")
        axis.grid(True, alpha=0.25)

    if not plotted:
        plt.close(fig)
        raise FigureGenerationError("No dimension prediction columns found.")

    fig.suptitle("Object Dimension Prediction", y=0.98)
    return _save_current(output_dir / "dimension_true_vs_pred.png", rect=(0.0, 0.0, 1.0, 0.92))


def plot_per_class_error(rows: list[dict[str, str]], output_dir: Path) -> Path | None:
    by_class: dict[str, list[float]] = {}

    for row in rows:
        class_name = row.get("semantic_class") or row.get("class_name") or row.get("label") or row.get("label_name") or "unknown"
        error = _float(row, "abs_error_distance_camera_m")
        if math.isfinite(error):
            by_class.setdefault(class_name, []).append(error)

    if len(by_class) < 2:
        return None

    means = sorted((sum(values) / len(values), name, len(values)) for name, values in by_class.items())
    labels = [f"{name}\n(n={count})" for _, name, count in means]
    values = [mean for mean, _, _ in means]

    plt.figure(figsize=(max(7.0, len(labels) * 0.75), 4.8))
    plt.bar(labels, values, color="#f28e2b", alpha=0.86)
    plt.ylabel("Mean Absolute Distance Error (m)")
    plt.title("Distance Error by Object Class")
    plt.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=35, ha="right")
    return _save_current(output_dir / "per_class_distance_error.png")


def _history_metric(entry: dict[str, Any], key: str) -> float | None:
    if key in entry:
        return float(entry[key])

    val_metrics = entry.get("val_metrics")
    if isinstance(val_metrics, dict) and key in val_metrics:
        return float(val_metrics[key])

    test_metrics = entry.get("test_metrics")
    if isinstance(test_metrics, dict) and key in test_metrics:
        return float(test_metrics[key])

    return None


def plot_training_curves(summary: dict[str, Any], output_dir: Path) -> Path | None:
    history = summary.get("history")
    if not isinstance(history, list) or not history:
        return None

    epochs = [int(entry.get("epoch", index + 1)) for index, entry in enumerate(history)]
    metrics = {
        "distance MAE": "mae_distance_camera_m",
        "position error": "mean_position_euclidean_error_m",
        "dimension error": "mean_dimension_euclidean_error_m",
    }

    plt.figure(figsize=(7.2, 4.8))
    plotted = False
    for label, key in metrics.items():
        values = [_history_metric(entry, key) for entry in history]
        if any(value is not None for value in values):
            plotted = True
            plt.plot(epochs, [float("nan") if value is None else value for value in values], marker="o", markersize=3, linewidth=1.5, label=label)

    if not plotted:
        plt.close()
        return None

    plt.xlabel("Epoch")
    plt.ylabel("Error (m)")
    plt.title("Validation Error During Training")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    return _save_current(output_dir / "training_curves.png")


def _nested_get(data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, (int, float)):
        return float(current)
    return None


def _first_metric(data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _nested_get(data, (key,))
        if value is not None:
            return value
    return None


def _comparison_metric_values(comparison: dict[str, Any]) -> dict[str, tuple[float, float]]:
    bbox = comparison.get("bbox_feature_baseline")
    candidate = comparison.get("candidate_model")
    if not isinstance(candidate, dict):
        candidate = comparison.get("crop_image_baseline")
    if not isinstance(bbox, dict) or not isinstance(candidate, dict):
        return {}

    metrics = {
        "Distance MAE": (
            _first_metric(bbox, ("distance_mae_m", "mae_distance_camera_m")),
            _first_metric(candidate, ("distance_mae_m", "mae_distance_camera_m")),
        ),
        "Position Error": (
            _first_metric(bbox, ("position_mean_euclidean_error_m", "mean_position_euclidean_error_m")),
            _first_metric(candidate, ("position_mean_euclidean_error_m", "mean_position_euclidean_error_m")),
        ),
        "Dimension Error": (
            _first_metric(bbox, ("dimensions_mean_euclidean_error_m", "mean_dimension_euclidean_error_m")),
            _first_metric(candidate, ("dimensions_mean_euclidean_error_m", "mean_dimension_euclidean_error_m")),
        ),
    }

    return {
        name: (float(bbox_value), float(crop_value))
        for name, (bbox_value, crop_value) in metrics.items()
        if bbox_value is not None and crop_value is not None
    }


def _backfill_crop_dimension_metric(comparison: dict[str, Any]) -> None:
    candidate = comparison.get("candidate_model")
    if not isinstance(candidate, dict):
        candidate = comparison.get("crop_image_baseline")
    if not isinstance(candidate, dict) or "dimensions_mean_euclidean_error_m" in candidate:
        return

    summary_path = comparison.get("crop_summary_path")
    if not isinstance(summary_path, str):
        return

    summary = _read_json(Path(summary_path))
    if not isinstance(summary, dict):
        return

    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        metrics = summary.get("test_metrics")

    if isinstance(metrics, dict) and isinstance(metrics.get("mean_dimension_euclidean_error_m"), (int, float)):
        candidate["dimensions_mean_euclidean_error_m"] = float(metrics["mean_dimension_euclidean_error_m"])


def _architecture_display_name(architecture: Any) -> str:
    names = {
        "rgb_crop_v2": "RGB Crop V2",
        "rgb_dual_context": "RGB Dual-Context",
        "depth_only": "Depth-Only",
        "rgbd_dual_context": "RGB-D Dual-Context",
        "masked_depth_geometry": "Masked Depth Geometry",
        "geometry_mlp": "Geometry Residual MLP",
        "geometry_rgbd": "Geometry + RGB-D Residual",
    }
    value = str(architecture or "candidate_model")
    return names.get(value, value.replace("_", " ").title())


def plot_baseline_comparison(comparison: dict[str, Any], output_dir: Path) -> Path | None:
    _backfill_crop_dimension_metric(comparison)
    values_by_metric = _comparison_metric_values(comparison)
    if not values_by_metric:
        return None

    labels = list(values_by_metric.keys())
    bbox_values = [values_by_metric[label][0] for label in labels]
    candidate_values = [values_by_metric[label][1] for label in labels]
    candidate_label = _architecture_display_name(
        comparison.get("candidate_architecture")
    )

    x_positions = range(len(labels))
    width = 0.34

    plt.figure(figsize=(7.2, 4.8))
    plt.bar([x - width / 2 for x in x_positions], bbox_values, width=width, label="BBox Features", color="#4e79a7")
    plt.bar([x + width / 2 for x in x_positions], candidate_values, width=width, label=candidate_label, color="#e15759")
    plt.xticks(list(x_positions), labels)
    plt.ylabel("Error (m)")
    plt.title("Spatial Prediction Baseline Comparison")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(frameon=False)
    return _save_current(output_dir / "baseline_comparison.png")


def _comparison_label(path: Path) -> str:
    name = path.parent.name
    prefix = "crop_spatial_baseline_"
    if name.startswith(prefix):
        name = name[len(prefix):]
    return name.replace("_", " ").title()


def _load_comparisons(paths: list[Path], labels: list[str] | None = None) -> list[tuple[str, dict[str, Any]]]:
    if labels is not None and len(labels) != len(paths):
        raise FigureGenerationError("--comparison-labels must match the number of --comparison-inputs")

    loaded: list[tuple[str, dict[str, Any]]] = []
    for index, path in enumerate(paths):
        comparison = _read_json(path)
        if comparison is None:
            continue
        _backfill_crop_dimension_metric(comparison)
        label = labels[index] if labels is not None else _comparison_label(path)
        loaded.append((label, comparison))
    return loaded


def plot_metric_comparison_suite(
    comparisons: list[tuple[str, dict[str, Any]]],
    output_dir: Path,
    metric_label: str,
    filename: str,
) -> Path | None:
    labels: list[str] = []
    candidate_values: list[float] = []
    bbox_value: float | None = None

    for label, comparison in comparisons:
        values = _comparison_metric_values(comparison).get(metric_label)
        if values is None:
            continue
        labels.append(label)
        if bbox_value is None:
            bbox_value = values[0]
        candidate_values.append(values[1])

    if not labels or bbox_value is None:
        return None

    bbox_label = "BBox + True Distance" if metric_label == "Dimension Error" else "BBox Features"
    display_labels = [bbox_label, *labels]
    values = [bbox_value, *candidate_values]
    colors = [
        "#4e79a7",
        "#f28e2b",
        "#59a14f",
        "#e15759",
        "#b07aa1",
        "#76b7b2",
        "#edc948",
        "#9c755f",
    ]
    x_positions = range(len(display_labels))

    plt.figure(figsize=(max(8.0, len(display_labels) * 1.35), 4.8))
    bars = plt.bar(
        list(x_positions),
        values,
        width=0.68,
        color=[colors[index % len(colors)] for index in range(len(values))],
        alpha=0.9,
    )
    plt.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)
    plt.xticks(list(x_positions), display_labels, rotation=20, ha="right")
    plt.ylabel("Error (m)")
    plt.title(f"{metric_label} by Input Architecture")
    plt.grid(True, axis="y", alpha=0.25)
    plt.ylim(0.0, max(values) * 1.18)
    return _save_current(output_dir / filename)


def plot_architecture_summary_suite(comparisons: list[tuple[str, dict[str, Any]]], output_dir: Path) -> Path | None:
    metric_labels = ["Distance MAE", "Position Error", "Dimension Error"]
    rows: list[tuple[str, list[float]]] = []

    for label, comparison in comparisons:
        metric_values = _comparison_metric_values(comparison)
        values = [metric_values[metric][1] for metric in metric_labels if metric in metric_values]
        if len(values) == len(metric_labels):
            rows.append((label, values))

    if not rows:
        return None

    x_positions = range(len(rows))
    width = 0.24
    colors = ["#4e79a7", "#59a14f", "#e15759"]

    plt.figure(figsize=(max(9.0, len(rows) * 1.35), 4.8))
    for offset, metric in enumerate(metric_labels):
        values = [row[1][offset] for row in rows]
        positions = [x + (offset - 1) * width for x in x_positions]
        plt.bar(positions, values, width=width, label=metric, color=colors[offset], alpha=0.86)

    plt.xticks(list(x_positions), [row[0] for row in rows], rotation=25, ha="right")
    plt.ylabel("Error (m)")
    plt.title("Spatial Error by Input Architecture")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(frameon=False)
    return _save_current(output_dir / "architecture_error_comparison.png")


def plot_relative_improvement_suite(comparisons: list[tuple[str, dict[str, Any]]], output_dir: Path) -> Path | None:
    labels: list[str] = []
    improvements: list[float] = []

    for label, comparison in comparisons:
        values = _comparison_metric_values(comparison).get("Position Error")
        if values is None or math.isclose(values[0], 0.0):
            continue
        bbox_value, crop_value = values
        labels.append(label)
        improvements.append((bbox_value - crop_value) / bbox_value * 100.0)

    if not labels:
        return None

    colors = ["#59a14f" if value >= 0 else "#e15759" for value in improvements]

    plt.figure(figsize=(max(8.0, len(labels) * 1.05), 4.8))
    plt.bar(labels, improvements, color=colors, alpha=0.86)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Position Error Reduction (%)")
    plt.title("Position Error Reduction Relative to BBox Features")
    plt.grid(True, axis="y", alpha=0.25)
    return _save_current(output_dir / "position_error_reduction_vs_bbox.png")


def generate_quantitative_comparison_figures(
    comparison_paths: list[Path],
    output_dir: Path,
    labels: list[str] | None = None,
) -> list[Path]:
    comparisons = _load_comparisons(comparison_paths, labels)
    if not comparisons:
        raise FigureGenerationError("No comparison JSON files were loaded.")

    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for metric_label, filename in (
        ("Distance MAE", "distance_baseline_comparison.png"),
        ("Position Error", "position_baseline_comparison.png"),
        ("Dimension Error", "dimension_baseline_comparison.png"),
    ):
        output = plot_metric_comparison_suite(comparisons, output_dir, metric_label, filename)
        if output is not None:
            outputs.append(output)

    architecture_summary = plot_architecture_summary_suite(comparisons, output_dir)
    if architecture_summary is not None:
        outputs.append(architecture_summary)

    improvement = plot_relative_improvement_suite(comparisons, output_dir)
    if improvement is not None:
        outputs.append(improvement)

    if len(comparisons) == 1:
        single = plot_baseline_comparison(comparisons[0][1], output_dir)
        if single is not None:
            outputs.append(single)

    return outputs


def generate_vision_figures(
    predictions_csv: Path,
    output_dir: Path,
    crop_summary_path: Path | None = None,
    comparison_path: Path | None = None,
) -> list[Path]:
    rows = _read_csv_rows(predictions_csv)
    if not rows:
        raise FigureGenerationError(f"No rows found in {predictions_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    outputs.append(plot_true_vs_pred_distance(rows, output_dir))
    outputs.append(plot_distance_error_histogram(rows, output_dir))
    outputs.append(plot_position_error_boxplot(rows, output_dir))
    outputs.append(plot_dimension_scatter(rows, output_dir))

    per_class = plot_per_class_error(rows, output_dir)
    if per_class is not None:
        outputs.append(per_class)

    summary = _read_json(crop_summary_path)
    if summary is not None:
        training_curve = plot_training_curves(summary, output_dir)
        if training_curve is not None:
            outputs.append(training_curve)

    comparison = _read_json(comparison_path)
    if comparison is not None:
        baseline_comparison = plot_baseline_comparison(comparison, output_dir)
        if baseline_comparison is not None:
            outputs.append(baseline_comparison)

    return outputs
