from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import subprocess
import sys
from typing import Any


@dataclass(frozen=True)
class PipelineStage:
    name: str
    command: list[str]
    expected_outputs: tuple[Path, ...]


def final_pipeline_config_fingerprint(config: dict[str, Any]) -> str:
    fingerprint_data = {
        key: value
        for key, value in config.items()
        if key != "config_path"
    }
    encoded = json.dumps(
        fingerprint_data,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_final_pipeline_config(
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    project_root = Path(project_root).expanduser().resolve()
    config_path = _resolved_path(project_root, config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Final pipeline config does not exist: {config_path}")

    with config_path.open() as file:
        config = json.load(file)

    pipeline_name = str(config.get("pipeline_name", "")).strip()
    if not pipeline_name:
        raise ValueError("pipeline_name is required in the final pipeline config.")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in pipeline_name):
        raise ValueError("pipeline_name may contain only letters, numbers, underscores, and hyphens.")

    raw_splits = config.get("raw_splits")
    if not isinstance(raw_splits, dict):
        raise ValueError("raw_splits must be a dictionary containing train, val, and test.")

    resolved_splits: dict[str, str] = {}
    for split in ("train", "val", "test"):
        if not raw_splits.get(split):
            raise ValueError(f"raw_splits.{split} is required.")
        raw_path = _resolved_path(project_root, raw_splits[split])
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw {split} dataset does not exist: {raw_path}")
        resolved_splits[split] = str(raw_path)

    config["pipeline_name"] = pipeline_name
    config["raw_splits"] = resolved_splits
    config["config_path"] = str(config_path)
    return config


def build_final_pipeline_stages(
    config: dict[str, Any],
    project_root: str | Path,
    python_executable: str | Path | None = None,
) -> list[PipelineStage]:
    root = Path(project_root).expanduser().resolve()
    python = str(python_executable or sys.executable)
    main = str(root / "main.py")
    name = config["pipeline_name"]
    raw = {split: Path(path) for split, path in config["raw_splits"].items()}

    image_size = int(config.get("image_size", 640))
    device = str(config.get("device", "mps"))
    segmentation_model = str(config.get("segmentation_model", "yolov8n-seg.pt"))
    segmentation_epochs = int(config.get("segmentation_epochs", 20))
    segmentation_batch = int(config.get("segmentation_batch", 8))
    segmentation_confidence = float(config.get("segmentation_confidence", 0.25))
    residual_epochs = int(config.get("residual_epochs", 100))
    residual_batch = int(config.get("residual_batch", 64))
    residual_image_size = int(config.get("residual_image_size", 128))

    crops = {
        split: root / "datasets" / "vision_crops" / f"{name}_{split}_masked"
        for split in raw
    }
    spatial = {
        split: root / "datasets" / "vision_labels" / f"object_spatial_features_{name}_{split}.csv"
        for split in raw
    }
    valid_labels = {
        split: raw[split] / "labels" / "vision_labels_valid.jsonl"
        for split in raw
    }
    audit_reports = {
        split: raw[split] / "labels" / "vision_labels_valid.audit.json"
        for split in raw
    }

    segmentation_dataset = root / "datasets" / f"tablesim_yolo_segmentation_{name}"
    segmentation_run_name = f"{name}_yolov8n_seg"
    segmentation_run = root / "YOLO" / "runs" / "detect" / segmentation_run_name
    segmentation_weights = segmentation_run / "weights" / "best.pt"
    segmentation_test_run_name = f"{name}_yolov8n_seg_test"
    segmentation_test_run = root / "YOLO" / "runs" / "detect" / segmentation_test_run_name

    predicted_crops = {
        split: root / "datasets" / "vision_crops" / f"{name}_{split}_predicted_masks"
        for split in raw
    }

    geometry: dict[str, dict[str, Path]] = {
        method: {
            split: root / "datasets" / "vision_geometry" / f"{name}_{split}_{method}"
            for split in raw
        }
        for method in ("oracle", "predicted_masks", "predicted_bbox")
    }

    residual_runs = {
        method: root / "YOLO" / "runs" / "detect" / f"{name}_geometry_{method}"
        for method in geometry
    }
    residual_predictions = {
        method: root / "predictions" / f"{name}_geometry_{method}_test"
        for method in geometry
    }

    baseline_report = root / "datasets" / "vision_labels" / f"vision_baseline_report_{name}.json"
    figure_root = root / "reports" / "figures"
    stages: list[PipelineStage] = []

    def command(*arguments: object) -> list[str]:
        return [python, main, *(str(argument) for argument in arguments)]

    for split in ("train", "val", "test"):
        raw_labels = raw[split] / "labels" / "vision_labels.jsonl"
        stages.append(PipelineStage(
            name=f"audit_{split}",
            command=command(
                "--audit-vision-dataset",
                "--vision-labels-input", raw_labels,
                "--unity-image-root", raw[split],
                "--filtered-vision-labels-output", valid_labels[split],
                "--vision-audit-report-output", audit_reports[split],
                "--audit-require-depth",
            ),
            expected_outputs=(valid_labels[split], audit_reports[split]),
        ))
        stages.append(PipelineStage(
            name=f"spatial_features_{split}",
            command=command(
                "--export-spatial-features",
                "--vision-labels-input", valid_labels[split],
                "--spatial-features-output", spatial[split],
            ),
            expected_outputs=(spatial[split],),
        ))
        stages.append(PipelineStage(
            name=f"masked_crops_{split}",
            command=command(
                "--export-vision-crops",
                "--vision-labels-input", valid_labels[split],
                "--unity-image-root", raw[split],
                "--vision-crops-output", crops[split],
                "--include-depth",
                "--include-target-mask",
            ),
            expected_outputs=(
                crops[split] / "crop_labels.csv",
                crops[split] / "frame_manifest.csv",
            ),
        ))

    stages.append(PipelineStage(
        name="bbox_feature_baseline",
        command=command(
            "--vision-baseline-report",
            "--spatial-features-train", spatial["train"],
            "--spatial-features-val", spatial["val"],
            "--spatial-features-test", spatial["test"],
            "--vision-baseline-report-output", baseline_report,
        ),
        expected_outputs=(baseline_report,),
    ))

    stages.append(PipelineStage(
        name="yolo_segmentation_dataset",
        command=command(
            "--export-yolo-segmentation",
            "--crop-labels-train", crops["train"] / "crop_labels.csv",
            "--crop-labels-val", crops["val"] / "crop_labels.csv",
            "--crop-labels-test", crops["test"] / "crop_labels.csv",
            "--yolo-segmentation-output", segmentation_dataset,
            "--clear-split",
        ),
        expected_outputs=(segmentation_dataset / "data.yaml",),
    ))

    stages.append(PipelineStage(
        name="train_yolo_segmentation",
        command=command(
            "--train-yolo",
            "--model", segmentation_model,
            "--yolo-data", segmentation_dataset / "data.yaml",
            "--run-name", segmentation_run_name,
            "--epochs", segmentation_epochs,
            "--imgsz", image_size,
            "--batch", segmentation_batch,
            "--device", device,
            "--yolo-exist-ok",
        ),
        expected_outputs=(segmentation_weights,),
    ))

    stages.append(PipelineStage(
        name="evaluate_yolo_segmentation_test",
        command=command(
            "--evaluate-yolo",
            "--weights", segmentation_weights,
            "--yolo-data", segmentation_dataset / "data.yaml",
            "--split", "test",
            "--eval-run-name", segmentation_test_run_name,
            "--imgsz", image_size,
            "--batch", segmentation_batch,
            "--device", device,
            "--yolo-exist-ok",
        ),
        expected_outputs=(segmentation_test_run / "metrics.json",),
    ))

    for split in ("train", "val", "test"):
        stages.append(PipelineStage(
            name=f"predicted_masks_{split}",
            command=command(
                "--export-predicted-mask-crops",
                "--crop-labels-input", crops[split] / "crop_labels.csv",
                "--weights", segmentation_weights,
                "--predicted-mask-output", predicted_crops[split],
                "--conf", segmentation_confidence,
                "--imgsz", image_size,
                "--device", device,
            ),
            expected_outputs=(
                predicted_crops[split] / "crop_labels.csv",
                predicted_crops[split] / "bbox_crop_labels.csv",
                predicted_crops[split] / "prediction_report.json",
            ),
        ))

    geometry_inputs = {
        "oracle": {split: crops[split] / "crop_labels.csv" for split in raw},
        "predicted_masks": {
            split: predicted_crops[split] / "crop_labels.csv" for split in raw
        },
        "predicted_bbox": {
            split: predicted_crops[split] / "bbox_crop_labels.csv" for split in raw
        },
    }

    for method in ("oracle", "predicted_masks", "predicted_bbox"):
        for split in ("train", "val", "test"):
            stages.append(PipelineStage(
                name=f"geometry_{method}_{split}",
                command=command(
                    "--evaluate-depth-geometry",
                    "--crop-labels-input", geometry_inputs[method][split],
                    "--depth-geometry-predictions-output", geometry[method][split] / "geometry.csv",
                    "--depth-geometry-report-output", geometry[method][split] / "report.json",
                ),
                expected_outputs=(
                    geometry[method][split] / "geometry.csv",
                    geometry[method][split] / "report.json",
                ),
            ))

        stages.append(PipelineStage(
            name=f"train_residual_{method}",
            command=command(
                "--train-geometry-residual",
                "--geometry-residual-architecture", "geometry_mlp",
                "--geometry-features-train", geometry[method]["train"] / "geometry.csv",
                "--geometry-features-val", geometry[method]["val"] / "geometry.csv",
                "--geometry-features-test", geometry[method]["test"] / "geometry.csv",
                "--geometry-residual-output", residual_runs[method],
                "--crop-spatial-epochs", residual_epochs,
                "--crop-spatial-batch", residual_batch,
                "--crop-spatial-image-size", residual_image_size,
            ),
            expected_outputs=(
                residual_runs[method] / "geometry_residual_best.pt",
                residual_runs[method] / "summary.json",
            ),
        ))

        stages.append(PipelineStage(
            name=f"export_residual_predictions_{method}",
            command=command(
                "--export-geometry-residual-predictions",
                "--geometry-features-test", geometry[method]["test"] / "geometry.csv",
                "--geometry-residual-checkpoint", residual_runs[method] / "geometry_residual_best.pt",
                "--geometry-residual-predictions-output", residual_predictions[method] / "predictions.csv",
                "--geometry-residual-predictions-summary", residual_predictions[method] / "prediction_summary.json",
                "--crop-spatial-batch", residual_batch,
            ),
            expected_outputs=(
                residual_predictions[method] / "predictions.csv",
                residual_predictions[method] / "prediction_summary.json",
            ),
        ))

        stages.append(PipelineStage(
            name=f"comparison_{method}",
            command=command(
                "--compare-vision-experiments",
                "--bbox-baseline-report", baseline_report,
                "--crop-baseline-summary", residual_runs[method] / "summary.json",
                "--vision-comparison-output", residual_runs[method] / "experiment_comparison.json",
            ),
            expected_outputs=(residual_runs[method] / "experiment_comparison.json",),
        ))

    stages.append(PipelineStage(
        name="segmentation_figures",
        command=command(
            "--make-segmentation-figures",
            "--ground-truth-crop-labels", crops["test"] / "crop_labels.csv",
            "--predicted-mask-crop-labels", predicted_crops["test"] / "crop_labels.csv",
            "--segmentation-figures-output", figure_root / f"{name}_predicted_segmentation_test",
        ),
        expected_outputs=(
            figure_root / f"{name}_predicted_segmentation_test" / "mask_metrics.json",
            figure_root / f"{name}_predicted_segmentation_test" / "predicted_mask_examples.png",
        ),
    ))

    stages.append(PipelineStage(
        name="predicted_mask_spatial_figures",
        command=command(
            "--make-vision-figures",
            "--predictions-csv", residual_predictions["predicted_masks"] / "predictions.csv",
            "--crop-summary-input", residual_runs["predicted_masks"] / "summary.json",
            "--comparison-input", residual_runs["predicted_masks"] / "experiment_comparison.json",
            "--figures-output", figure_root / f"{name}_geometry_predicted_masks",
        ),
        expected_outputs=(
            figure_root / f"{name}_geometry_predicted_masks" / "true_vs_pred_distance.png",
            figure_root / f"{name}_geometry_predicted_masks" / "dimension_true_vs_pred.png",
        ),
    ))

    stages.append(PipelineStage(
        name="three_way_comparison_figures",
        command=command(
            "--make-quantitative-figures",
            "--comparison-inputs",
            residual_runs["oracle"] / "experiment_comparison.json",
            residual_runs["predicted_masks"] / "experiment_comparison.json",
            residual_runs["predicted_bbox"] / "experiment_comparison.json",
            "--comparison-labels", "Oracle Mask", "Predicted Mask", "Predicted Bbox",
            "--quantitative-figures-output", figure_root / f"{name}_mask_realism_comparison",
        ),
        expected_outputs=(
            figure_root / f"{name}_mask_realism_comparison" / "architecture_error_comparison.png",
        ),
    ))

    return stages


def _outputs_exist(stage: PipelineStage) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in stage.expected_outputs)


def _write_final_summary(
    config: dict[str, Any],
    project_root: Path,
    output_path: Path,
) -> None:
    name = config["pipeline_name"]

    def read_json(path: Path) -> dict[str, Any]:
        with path.open() as file:
            return json.load(file)

    summary = {
        "pipeline_name": name,
        "raw_splits": config["raw_splits"],
        "audit_reports": {},
        "segmentation_test": read_json(
            project_root / "YOLO" / "runs" / "detect"
            / f"{name}_yolov8n_seg_test" / "metrics.json"
        ),
        "geometry_residual_test": {},
    }

    for split, raw_path in config["raw_splits"].items():
        report = Path(raw_path) / "labels" / "vision_labels_valid.audit.json"
        summary["audit_reports"][split] = read_json(report)

    for method in ("oracle", "predicted_masks", "predicted_bbox"):
        model_summary = read_json(
            project_root / "YOLO" / "runs" / "detect"
            / f"{name}_geometry_{method}" / "summary.json"
        )
        summary["geometry_residual_test"][method] = model_summary["test_metrics"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")


def run_final_vision_pipeline(
    config_path: str | Path,
    project_root: str | Path,
    restart: bool = False,
) -> Path:
    root = Path(project_root).expanduser().resolve()
    config = load_final_pipeline_config(config_path, root)
    stages = build_final_pipeline_stages(config, root)
    config_fingerprint = final_pipeline_config_fingerprint(config)
    state_dir = root / "reports" / "pipelines" / config["pipeline_name"]
    state_path = state_dir / "pipeline_state.json"
    summary_path = state_dir / "final_summary.json"
    state_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "pipeline_name": config["pipeline_name"],
        "config_path": config["config_path"],
        "config_fingerprint": config_fingerprint,
        "raw_splits": config["raw_splits"],
        "status": "pending",
        "current_stage": None,
        "current_stage_index": 0,
        "total_stages": len(stages),
        "completed_stages": [],
        "failed_stage": None,
    }
    if state_path.exists() and not restart:
        with state_path.open() as file:
            state = json.load(file)

        if state.get("config_fingerprint") != config_fingerprint:
            raise ValueError(
                "The saved pipeline progress belongs to different datasets or "
                "settings. Use a new pipeline_name or rerun with "
                "--pipeline-restart."
            )

    completed = set(state.get("completed_stages", []))

    for index, stage in enumerate(stages, start=1):
        if stage.name in completed and _outputs_exist(stage):
            print(
                f"[{index}/{len(stages)}] Skipping completed stage: {stage.name}",
                flush=True,
            )
            continue

        state["status"] = "running"
        state["current_stage"] = stage.name
        state["current_stage_index"] = index
        state["total_stages"] = len(stages)
        state["failed_stage"] = None
        state_path.write_text(json.dumps(state, indent=2) + "\n")

        print(f"[{index}/{len(stages)}] Running stage: {stage.name}", flush=True)
        print(" ".join(stage.command), flush=True)

        try:
            subprocess.run(stage.command, cwd=root, check=True)
        except subprocess.CalledProcessError:
            state["status"] = "failed"
            state["failed_stage"] = stage.name
            state["completed_stages"] = [
                item.name for item in stages if item.name in completed
            ]
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            raise

        if not _outputs_exist(stage):
            state["status"] = "failed"
            state["failed_stage"] = stage.name
            state["completed_stages"] = [
                item.name for item in stages if item.name in completed
            ]
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            raise RuntimeError(
                f"Stage '{stage.name}' completed but expected outputs are missing: "
                f"{stage.expected_outputs}"
            )

        completed.add(stage.name)
        state["current_stage"] = None
        state["failed_stage"] = None
        state["completed_stages"] = [
            item.name for item in stages if item.name in completed
        ]
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        print(f"[{index}/{len(stages)}] Completed stage: {stage.name}", flush=True)

    _write_final_summary(config, root, summary_path)
    state["status"] = "complete"
    state["current_stage"] = None
    state["current_stage_index"] = len(stages)
    state["final_summary"] = str(summary_path)
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(f"Final vision pipeline complete. Summary: {summary_path}", flush=True)
    return summary_path
