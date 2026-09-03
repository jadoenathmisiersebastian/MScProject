from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.yolo_conversion import convert_dataset
from src.yolo_prediction import predict_yolo
from src.yolo_training import train_yolo
from src.yolo_evaluation import evaluate_yolo
from src.yolo_segmentation_dataset import export_yolo_segmentation_dataset
from src.yolo_rgbd_dataset import export_rgbd_segmentation_dataset
from src.yolo_rgbd_prediction import predict_rgbd_segmentation
from src.yolo_depth_verified_prediction import predict_depth_verified_segmentation
from src.vision_predicted_masks import export_predicted_mask_crops
from src.final_vision_pipeline import run_final_vision_pipeline

from src.detection_output import run_detection_export
from src.gs_pose_selection import run_pose_selection
from src.vision_output import run_vision_output_export
from src.vision_semantic_enrichment import enrich_vision_labels_with_semantic_classes
from src.vision_dataset_audit import audit_and_filter_vision_dataset
from src.vision_spatial_features import export_object_spatial_features
from src.vision_distance_baseline import train_distance_baseline
from src.vision_position_baseline import train_position_baseline
from src.vision_dimensions_baseline import train_dimensions_baseline
from src.vision_baseline_report import (
    run_vision_baseline_report,
    run_vision_baseline_report_from_splits,
)
from src.vision_crop_dataset import export_vision_crops
from src.vision_depth_geometry import evaluate_depth_geometry
from src.vision_geometry_residual import (
    GEOMETRY_RESIDUAL_ARCHITECTURES,
    export_geometry_residual_predictions,
    train_geometry_residual_from_splits,
)
from src.vision_experiment_compare import compare_vision_experiments

from src.vision_crop_spatial_baseline import (
    train_crop_spatial_baseline,
    train_crop_spatial_baseline_from_splits,
)

from src.vision_crop_spatial_predict import (
    export_crop_spatial_predictions,
    summarize_prediction_errors,
)
from src.vision_multimodal_spatial import (
    SPATIAL_ARCHITECTURES,
    export_multimodal_spatial_predictions,
    train_multimodal_spatial_from_splits,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "paths.yaml"


def parse_ratio(values: list[str] | None) -> tuple[float, float, float] | None:
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError("--split-ratio expects exactly three values: train val test")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Object detection pipeline for Unity synthetic data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    actions = parser.add_argument_group("actions")
    actions.add_argument("--yolo-conversion", action="store_true", help="Convert Unity SOLO output to YOLO format.")
    actions.add_argument("--train-yolo", action="store_true", help="Train a YOLO detector.")
    actions.add_argument("--evaluate-yolo", action="store_true", help="Evaluate a YOLO detector on an explicit dataset split.")
    actions.add_argument("--predict-yolo", action="store_true", help="Run YOLO prediction and draw boxes.")
    actions.add_argument("--export-yolo-segmentation", action="store_true", help="Convert Unity target masks into an explicit-split YOLO segmentation dataset.")
    actions.add_argument("--export-rgbd-yolo-segmentation", action="store_true", help="Create a four-channel RGB-D YOLO segmentation dataset from aligned Unity depth.")
    actions.add_argument("--predict-rgbd-yolo", action="store_true", help="Run a four-channel YOLO segmentation checkpoint on one aligned RGB-D frame.")
    actions.add_argument("--predict-depth-verified-yolo", action="store_true", help="Generate RGB YOLO proposals and reject only strong metric-depth contradictions.")
    actions.add_argument("--export-predicted-mask-crops", action="store_true", help="Apply YOLO-predicted instance masks to aligned depth crops.")
    actions.add_argument("--run-final-vision-pipeline", action="store_true", help="Run the complete resumable final RGB-D vision pipeline from a three-split config.")
    actions.add_argument("--pipeline-ui", action="store_true", help="Open the desktop launcher for the final RGB-D vision pipeline.")

    actions.add_argument("--export-detections", action="store_true", help="Export YOLO detections to structured JSON.")
    actions.add_argument("--select-pose", action="store_true", help="Run focus-based grasp/pose selection.")
    actions.add_argument("--export-vision-output", action="store_true", help="Export richer vision output from detections JSON.")
    actions.add_argument("--enrich-vision-semantic-labels", action="store_true", help="Add Perception semantic classes to existing rich vision labels.")
    actions.add_argument("--audit-vision-dataset", action="store_true", help="Audit raw Unity captures and write a filtered valid vision-label JSONL.")
    actions.add_argument("--export-spatial-features", action="store_true", help="Export object-level spatial feature CSV from synthetic vision labels.")
    actions.add_argument("--train-distance-baseline", action="store_true", help="Train/evaluate a simple distance-from-bbox baseline.")
    actions.add_argument("--train-position-baseline", action="store_true", help="Train/evaluate a simple camera-relative position baseline.")
    actions.add_argument("--train-dimensions-baseline", action="store_true", help="Train/evaluate a simple object-dimensions baseline.")
    actions.add_argument("--vision-baseline-report", action="store_true", help="Run all simple vision spatial baselines and save one report.")
    actions.add_argument("--export-vision-crops", action="store_true", help="Export object crop images and labels from rich synthetic vision labels.")
    actions.add_argument("--evaluate-depth-geometry", action="store_true", help="Evaluate calibrated target-mask depth geometry against spatial labels.")
    actions.add_argument("--train-geometry-residual", action="store_true", help="Train a calibrated-geometry residual correction model.")
    actions.add_argument("--export-geometry-residual-predictions", action="store_true", help="Export corrected geometry predictions from a residual model.")
    actions.add_argument("--train-crop-spatial-baseline", action="store_true", help="Train/evaluate an image-crop spatial regression baseline.")
    actions.add_argument("--train-multimodal-spatial", action="store_true", help="Train/evaluate a dual-context, depth-only, or RGB-D spatial model.")
    actions.add_argument("--compare-vision-experiments", action="store_true", help="Compare bbox-feature and crop-image spatial baselines.")
    actions.add_argument("--export-crop-spatial-predictions", action="store_true", help="Export crop spatial model predictions to CSV.")
    actions.add_argument("--export-multimodal-spatial-predictions", action="store_true", help="Export multimodal spatial model predictions to CSV.")

    conversion = parser.add_argument_group("conversion")
    conversion.add_argument("--dataset", help="Optional Unity dataset folder name/path. Overrides unity_dataset_path in paths.yaml.")
    conversion.add_argument("--split", choices=("train", "val", "test"), default="train")
    conversion.add_argument("--split-ratio", nargs=3, metavar=("TRAIN", "VAL", "TEST"), help="Randomly split one dataset, e.g. --split-ratio 0.8 0.1 0.1")
    conversion.add_argument("--seed", type=int, default=42)
    conversion.add_argument("--clear-split", action="store_true", help="Clear the selected output split before converting.")
    conversion.add_argument("--yolo-segmentation-output", type=Path, help="Output root for the YOLO segmentation dataset.")
    conversion.add_argument("--segmentation-polygon-tolerance", type=float, default=0.001, help="Contour simplification tolerance as a fraction of perimeter.")
    conversion.add_argument("--segmentation-minimum-component-fraction", type=float, default=0.75, help="Minimum mask area retained by the largest connected component.")
    conversion.add_argument("--rgbd-source-yolo-dataset", type=Path, help="Existing RGB YOLO segmentation dataset whose images and labels should be reused.")
    conversion.add_argument("--rgbd-segmentation-output", type=Path, help="Output root for the four-channel RGB-D YOLO dataset.")
    conversion.add_argument("--rgbd-manifest-train", type=Path, help="Training frame_manifest.csv containing source Unity image paths.")
    conversion.add_argument("--rgbd-manifest-val", type=Path, help="Validation frame_manifest.csv containing source Unity image paths.")
    conversion.add_argument("--rgbd-manifest-test", type=Path, help="Test frame_manifest.csv containing source Unity image paths.")
    conversion.add_argument("--rgbd-export-long-side", type=int, default=640, help="Stored RGB-D tensor long-side resolution.")
    conversion.add_argument("--rgbd-export-limit", type=int, help="Optional per-split image limit for smoke tests.")

    training = parser.add_argument_group("training")
    training.add_argument("--model", help="Base YOLO model, e.g. yolov8n.pt")
    training.add_argument("--run-name", help="Training run name.")
    training.add_argument("--epochs", type=int)
    training.add_argument("--imgsz", type=int)
    training.add_argument("--batch", type=int)
    training.add_argument("--device", help="Training device, e.g. cpu, mps, 0.")
    training.add_argument("--eval-run-name", help="Output run name for YOLO evaluation.")
    training.add_argument("--yolo-data", type=Path, help="Optional data.yaml override for YOLO training or evaluation.")
    training.add_argument("--yolo-exist-ok", action="store_true", help="Allow YOLO to reuse the requested run directory.")

    pipeline = parser.add_argument_group("final pipeline")
    pipeline.add_argument("--pipeline-config", type=Path, default=Path("config/final_vision_pipeline.json"), help="JSON config containing final train/val/test raw dataset paths.")
    pipeline.add_argument("--pipeline-restart", action="store_true", help="Ignore saved pipeline stage state and rerun every stage.")

    prediction = parser.add_argument_group("prediction")
    prediction.add_argument("--weights", type=Path, help="Path to trained best.pt.")
    prediction.add_argument("--input-dir", type=Path, help="Image folder for prediction.")
    prediction.add_argument("--output-dir", type=Path, help="Prediction output folder.")
    prediction.add_argument("--rgb-input", type=Path, help="RGB image for four-channel prediction.")
    prediction.add_argument("--depth-input", type=Path, help="Aligned metric-depth EXR or NPY for four-channel prediction.")
    prediction.add_argument("--metadata-input", type=Path, help="Processed real-frame metadata JSON containing calibrated depth intrinsics.")
    prediction.add_argument("--depth-verifier-output", type=Path, help="Optional JSON output for depth-verified RGB proposals.")
    prediction.add_argument("--conf", type=float, default=0.25)
    prediction.add_argument("--detections-output", type=Path, help="Output path for detections JSON.")
    prediction.add_argument("--detections-input", type=Path, help="Input detections JSON for pose selection.")
    prediction.add_argument("--pose-output", type=Path, help="Output path for pose selection JSON.")
    prediction.add_argument("--vision-output", type=Path, help="Output path for richer vision output JSON.")
    prediction.add_argument("--min-confidence", type=float, default=0.25)
    prediction.add_argument("--max-focus-distance", type=float, default=0.25)
    prediction.add_argument("--vision-labels-input", type=Path, help="Input synthetic vision labels JSONL.")
    prediction.add_argument("--vision-labels-output", type=Path, help="Output enriched synthetic vision labels JSONL.")
    prediction.add_argument("--semantic-minimum-iou", type=float, default=0.05, help="Minimum bbox IoU for multi-object semantic matching.")
    prediction.add_argument("--filtered-vision-labels-output", type=Path, help="Output JSONL containing only frames that pass the dataset audit.")
    prediction.add_argument("--vision-audit-report-output", type=Path, help="Output JSON report for the vision dataset audit.")
    prediction.add_argument("--audit-minimum-iou", type=float, default=0.05, help="Minimum IoU between rich and Perception bboxes during audit.")
    prediction.add_argument("--audit-minimum-visible-pixels", type=int, default=1, help="Minimum non-background semantic pixels required during audit.")
    prediction.add_argument("--audit-require-depth", action="store_true", help="Require exactly one existing depth image per audited frame.")
    prediction.add_argument("--spatial-features-output", type=Path, help="Output object spatial features CSV.")
    prediction.add_argument("--spatial-features-input", type=Path, help="Input object spatial features CSV.")
    prediction.add_argument("--spatial-features-train", type=Path, help="Train object spatial features CSV for explicit split baseline reports.")
    prediction.add_argument("--spatial-features-val", type=Path, help="Validation object spatial features CSV for explicit split baseline reports.")
    prediction.add_argument("--spatial-features-test", type=Path, help="Test object spatial features CSV for explicit split baseline reports.")
    prediction.add_argument("--vision-baseline-report-output", type=Path, help="Output path for vision baseline report JSON.")
    prediction.add_argument("--unity-image-root", type=Path, help="Unity Perception dataset root containing sequence.0 images.")
    prediction.add_argument("--vision-crops-output", type=Path, help="Output directory for object crop dataset.")
    prediction.add_argument("--crop-padding", type=float, default=0.10, help="Fractional bbox padding for object crops.")
    prediction.add_argument("--include-depth", action="store_true", help="Export aligned metric depth crops from SOLO DepthAnnotation EXR files.")
    prediction.add_argument("--include-target-mask", action="store_true", help="Export target semantic masks and target-only masked depth crops.")
    prediction.add_argument("--predicted-mask-output", type=Path, help="Output directory for predicted masks, masked depth, and crop labels.")
    prediction.add_argument("--predicted-mask-minimum-iou", type=float, default=0.10, help="Minimum bbox IoU used to associate a predicted mask with an evaluation target.")
    prediction.add_argument("--predicted-mask-depth-mad-scale", type=float, default=3.0, help="Robust MAD multiplier used to reject background depth leaking through predicted-mask boundaries.")
    prediction.add_argument("--predicted-mask-depth-band-min", type=float, default=0.08, help="Minimum predicted-mask depth support band in metres.")
    prediction.add_argument("--predicted-mask-depth-band-max", type=float, default=0.25, help="Maximum predicted-mask depth support band in metres.")
    prediction.add_argument("--depth-geometry-predictions-output", type=Path, help="Output CSV for per-object calibrated depth geometry estimates.")
    prediction.add_argument("--depth-geometry-report-output", type=Path, help="Output JSON report for calibrated depth geometry errors.")
    prediction.add_argument("--geometry-lower-quantile", type=float, default=0.01, help="Lower point-cloud quantile for the robust geometry estimate.")
    prediction.add_argument("--geometry-upper-quantile", type=float, default=0.99, help="Upper point-cloud quantile for the robust geometry estimate.")
    prediction.add_argument("--geometry-features-train", type=Path, help="Training calibrated-geometry prediction CSV.")
    prediction.add_argument("--geometry-features-val", type=Path, help="Validation calibrated-geometry prediction CSV.")
    prediction.add_argument("--geometry-features-test", type=Path, help="Test calibrated-geometry prediction CSV.")
    prediction.add_argument("--geometry-residual-architecture", choices=GEOMETRY_RESIDUAL_ARCHITECTURES, help="Geometry residual model architecture.")
    prediction.add_argument("--geometry-residual-output", type=Path, help="Output directory for geometry residual training.")
    prediction.add_argument("--geometry-residual-checkpoint", type=Path, help="Geometry residual checkpoint path.")
    prediction.add_argument("--geometry-residual-predictions-output", type=Path, help="Output CSV for corrected geometry predictions.")
    prediction.add_argument("--geometry-residual-predictions-summary", type=Path, help="Output JSON summary for corrected geometry predictions.")
    prediction.add_argument("--crop-labels-input", type=Path, help="Input crop labels CSV.")
    prediction.add_argument("--crop-spatial-output", type=Path, help="Output directory for crop spatial baseline.")
    prediction.add_argument("--crop-spatial-epochs", type=int, default=50, help="Epochs for crop spatial baseline.")
    prediction.add_argument("--crop-spatial-batch", type=int, default=16, help="Batch size for crop spatial baseline.")
    prediction.add_argument("--crop-spatial-image-size", type=int, default=128, help="Input crop image size.")
    prediction.add_argument("--spatial-architecture", choices=SPATIAL_ARCHITECTURES, help="Multimodal spatial architecture.")
    prediction.add_argument("--multimodal-spatial-output", type=Path, help="Output directory for a multimodal spatial model.")
    prediction.add_argument("--context-image-size", type=int, default=128, help="Full-scene context image size.")
    prediction.add_argument("--depth-min-m", type=float, default=0.3, help="Minimum metric depth used for fixed depth normalization.")
    prediction.add_argument("--depth-max-m", type=float, default=3.0, help="Maximum metric depth used for fixed depth normalization.")
    prediction.add_argument("--bbox-baseline-report", type=Path, help="Input bbox-feature baseline report JSON.")
    prediction.add_argument("--crop-baseline-summary", type=Path, help="Input crop-image baseline summary JSON.")
    prediction.add_argument("--vision-comparison-output", type=Path, help="Output comparison JSON.")
    prediction.add_argument("--crop-labels-train", type=Path, help="Train crop labels CSV for explicit split training.")
    prediction.add_argument("--crop-labels-val", type=Path, help="Validation crop labels CSV for explicit split training.")
    prediction.add_argument("--crop-labels-test", type=Path, help="Test crop labels CSV for explicit split training.")
    prediction.add_argument("--crop-spatial-checkpoint", type=Path, help="Crop spatial model checkpoint path.")
    prediction.add_argument("--crop-spatial-predictions-output", type=Path, help="Output CSV for crop spatial predictions.")
    prediction.add_argument("--crop-spatial-predictions-summary", type=Path, help="Output JSON summary for crop spatial prediction errors.")
    prediction.add_argument("--multimodal-spatial-checkpoint", type=Path, help="Multimodal spatial checkpoint path.")
    prediction.add_argument("--multimodal-spatial-predictions-output", type=Path, help="Output CSV for multimodal spatial predictions.")
    prediction.add_argument("--multimodal-spatial-predictions-summary", type=Path, help="Output JSON summary for multimodal spatial prediction errors.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.pipeline_ui:
        from src.final_vision_pipeline_ui import launch_final_pipeline_ui

        launch_final_pipeline_ui(
            config_path=args.pipeline_config,
            project_root=Path(__file__).resolve().parent,
        )
        return

    config = load_config(args.config)

    if args.yolo_conversion:
        summary = convert_dataset(
            config=config,
            dataset=args.dataset,
            split=args.split,
            ratios=parse_ratio(args.split_ratio),
            seed=args.seed,
            clear_split=args.clear_split,
        )
        print("YOLO conversion complete")
        print(f"Dataset: {summary.dataset_path}")
        print(f"Output:  {summary.output_path}")
        print(f"Counts:  {summary.split_counts}")
        print(f"Skipped empty frames: {summary.skipped_empty}")

    if args.export_yolo_segmentation:
        if args.crop_labels_train is None or args.crop_labels_val is None or args.crop_labels_test is None:
            raise ValueError(
                "--crop-labels-train, --crop-labels-val, and --crop-labels-test "
                "are required for --export-yolo-segmentation"
            )
        output_path = args.yolo_segmentation_output
        if output_path is None:
            output_path = config.project_code / "datasets" / "tablesim_yolo_segmentation"

        summaries = export_yolo_segmentation_dataset(
            crop_label_splits=(
                ("train", args.crop_labels_train),
                ("val", args.crop_labels_val),
                ("test", args.crop_labels_test),
            ),
            output_path=output_path,
            classes=config.classes,
            simplification_tolerance=args.segmentation_polygon_tolerance,
            minimum_component_fraction=args.segmentation_minimum_component_fraction,
            clear_splits=args.clear_split,
        )
        print("YOLO segmentation export complete:")
        for split_name, summary in summaries.items():
            print(
                f"{split_name}: images={summary.split_counts[split_name]}, "
                f"objects={summary.object_counts[split_name]}, "
                f"negative_frames={summary.negative_frame_count}, "
                f"skipped={summary.skipped_objects}, "
                f"fragmented={summary.fragmented_masks}"
            )
        print(f"Output: {Path(output_path).expanduser().resolve()}")

    if args.export_rgbd_yolo_segmentation:
        required = {
            "--rgbd-source-yolo-dataset": args.rgbd_source_yolo_dataset,
            "--rgbd-manifest-train": args.rgbd_manifest_train,
            "--rgbd-manifest-val": args.rgbd_manifest_val,
            "--rgbd-manifest-test": args.rgbd_manifest_test,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "Required for --export-rgbd-yolo-segmentation: "
                + ", ".join(missing)
            )
        output_path = args.rgbd_segmentation_output
        if output_path is None:
            output_path = config.project_code / "datasets" / "tablesim_yolo_segmentation_rgbd"
        summaries = export_rgbd_segmentation_dataset(
            source_dataset=args.rgbd_source_yolo_dataset,
            manifest_splits=(
                ("train", args.rgbd_manifest_train),
                ("val", args.rgbd_manifest_val),
                ("test", args.rgbd_manifest_test),
            ),
            output_path=output_path,
            long_side=args.rgbd_export_long_side,
            depth_min_m=args.depth_min_m,
            depth_max_m=args.depth_max_m,
            clear_splits=args.clear_split,
            limit_per_split=args.rgbd_export_limit,
        )
        print("RGB-D YOLO segmentation export complete:")
        for split_name, summary in summaries.items():
            print(
                f"{split_name}: images={summary.num_images}, "
                f"negative_frames={summary.num_negative_images}, "
                f"written={summary.num_written}, reused={summary.num_reused}, "
                f"valid_depth={summary.mean_valid_depth_fraction:.4f}"
            )
        print(f"Output: {Path(output_path).expanduser().resolve()}")

    if args.train_yolo:
        train_yolo(
            config=config,
            model=args.model,
            run_name=args.run_name,
            epochs=args.epochs,
            image_size=args.imgsz,
            batch=args.batch,
            device=args.device,
            data_yaml=args.yolo_data,
            exist_ok=args.yolo_exist_ok,
        )

    if args.run_final_vision_pipeline:
        output = run_final_vision_pipeline(
            config_path=args.pipeline_config,
            project_root=Path(__file__).resolve().parent,
            restart=args.pipeline_restart,
        )
        print(f"Final pipeline summary saved to: {output}")

    if args.evaluate_yolo:
        output = evaluate_yolo(
            config=config,
            weights=args.weights,
            split=args.split,
            run_name=args.eval_run_name,
            image_size=args.imgsz,
            batch=args.batch,
            device=args.device,
            data_yaml=args.yolo_data,
            exist_ok=args.yolo_exist_ok,
        )
        print(f"YOLO {args.split} evaluation metrics saved to: {output}")

    if args.predict_yolo:
        output = predict_yolo(
            config=config,
            split=args.split,
            weights=args.weights,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            conf=args.conf,
        )
        print(f"Predictions saved to: {output}")

    if args.predict_rgbd_yolo:
        if args.weights is None or args.rgb_input is None or args.depth_input is None:
            raise ValueError(
                "--weights, --rgb-input, and --depth-input are required for "
                "--predict-rgbd-yolo"
            )
        result = predict_rgbd_segmentation(
            weights=args.weights,
            rgb_path=args.rgb_input,
            depth_path=args.depth_input,
            image_size=args.imgsz or config.image_size,
            confidence=args.conf,
            device=args.device,
            depth_min_m=args.depth_min_m,
            depth_max_m=args.depth_max_m,
        )
        count = len(result.boxes) if result.boxes is not None else 0
        print(f"RGB-D detections: {count}")

    if args.predict_depth_verified_yolo:
        required = {
            "--weights": args.weights,
            "--rgb-input": args.rgb_input,
            "--depth-input": args.depth_input,
            "--metadata-input": args.metadata_input,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "Required for --predict-depth-verified-yolo: " + ", ".join(missing)
            )
        output = predict_depth_verified_segmentation(
            weights=args.weights,
            rgb_path=args.rgb_input,
            depth_path=args.depth_input,
            metadata_path=args.metadata_input,
            proposal_confidence=args.conf,
            image_size=args.imgsz or config.image_size,
            device=args.device,
            maximum_focus_distance=args.max_focus_distance,
        )
        serializable = {key: value for key, value in output.items() if key != "result"}
        if args.depth_verifier_output is not None:
            destination = args.depth_verifier_output.expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(serializable, indent=2))
            print(f"Depth-verified proposals saved to: {destination}")
        focused = serializable["focused_candidate"]
        print(
            "Focused depth-verified prediction: "
            + ("none" if focused is None else f"{focused['class_name']} ({focused['confidence']:.3f})")
        )


    if args.export_detections:
        output = run_detection_export(
            config=config,
            split=args.split,
            weights=args.weights,
            input_dir=args.input_dir,
            output_path=args.detections_output,
            conf=args.conf,
        )
        print(f"Detections saved to: {output}")

    if args.select_pose:
        detections_path = args.detections_input

        if detections_path is None:
            detections_path = config.predictions_output / args.split / "detections.json"

        output = run_pose_selection(
            detections_path=detections_path,
            output_path=args.pose_output,
            min_confidence=args.min_confidence,
            max_focus_distance=args.max_focus_distance,
        )
        print(f"Pose selections saved to: {output}")


    if args.export_vision_output:
        detections_path = args.detections_input

        if detections_path is None:
            detections_path = config.predictions_output / args.split / "detections.json"

        output = run_vision_output_export(
            detections_path=detections_path,
            output_path=args.vision_output,
            min_confidence=args.min_confidence,
            max_focus_distance=args.max_focus_distance,
        )

        print(f"Vision output saved to: {output}")

    if args.enrich_vision_semantic_labels:
        if args.vision_labels_input is None:
            raise ValueError("--vision-labels-input is required for --enrich-vision-semantic-labels")
        if args.unity_image_root is None:
            raise ValueError("--unity-image-root is required for --enrich-vision-semantic-labels")

        output_path = args.vision_labels_output
        if output_path is None:
            output_path = args.vision_labels_input.with_name(
                f"{args.vision_labels_input.stem}_semantic.jsonl"
            )

        summary = enrich_vision_labels_with_semantic_classes(
            vision_labels_path=args.vision_labels_input,
            dataset_root=args.unity_image_root,
            output_path=output_path,
            minimum_iou=args.semantic_minimum_iou,
        )

        print("Semantic label enrichment complete:")
        for key, value in summary.items():
            print(f"{key}: {value}")

    if args.audit_vision_dataset:
        if args.vision_labels_input is None:
            raise ValueError("--vision-labels-input is required for --audit-vision-dataset")
        if args.unity_image_root is None:
            raise ValueError("--unity-image-root is required for --audit-vision-dataset")

        filtered_output_path = args.filtered_vision_labels_output
        if filtered_output_path is None:
            filtered_output_path = args.vision_labels_input.with_name(
                f"{args.vision_labels_input.stem}_valid.jsonl"
            )

        report_output_path = args.vision_audit_report_output
        if report_output_path is None:
            report_output_path = filtered_output_path.with_suffix(".audit.json")

        report = audit_and_filter_vision_dataset(
            vision_labels_path=args.vision_labels_input,
            dataset_root=args.unity_image_root,
            filtered_output_path=filtered_output_path,
            report_output_path=report_output_path,
            minimum_iou=args.audit_minimum_iou,
            minimum_visible_pixels=args.audit_minimum_visible_pixels,
            require_depth=args.audit_require_depth,
        )

        print("Vision dataset audit complete:")
        print(f"Input frames: {report['num_input_frames']}")
        print(f"Valid frames: {report['num_valid_frames']}")
        print(f"Invalid frames: {report['num_invalid_frames']}")
        print(f"Issue counts: {report['issue_counts']}")
        print(f"Filtered labels: {filtered_output_path}")
        print(f"Audit report: {report_output_path}")

    if args.export_spatial_features:
        vision_labels_path = args.vision_labels_input

        if vision_labels_path is None:
            vision_labels_path = config.project_code / "datasets" / "vision_labels" / "vision_labels_debug.jsonl"

        output_path = args.spatial_features_output

        if output_path is None:
            output_path = config.project_code / "datasets" / "vision_labels" / "object_spatial_features.csv"

        output = export_object_spatial_features(
            vision_labels_path=vision_labels_path,
            output_path=output_path,
            validate=True,
        )

        print(f"Object spatial features saved to: {output}")

    if args.train_distance_baseline:
        spatial_features_path = args.spatial_features_input

        if spatial_features_path is None:
            spatial_features_path = config.project_code / "datasets" / "vision_labels" / "object_spatial_features.csv"

        summary = train_distance_baseline(
            csv_path=spatial_features_path,
            test_ratio=0.2,
            seed=args.seed,
        )

        print("Distance baseline summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")

    if args.train_position_baseline:
        spatial_features_path = args.spatial_features_input

        if spatial_features_path is None:
            spatial_features_path = config.project_code / "datasets" / "vision_labels" / "object_spatial_features.csv"

        summary = train_position_baseline(
            csv_path=spatial_features_path,
            test_ratio=0.2,
            seed=args.seed,
        )

        print("Position baseline summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")

    if args.train_dimensions_baseline:
        spatial_features_path = args.spatial_features_input

        if spatial_features_path is None:
            spatial_features_path = config.project_code / "datasets" / "vision_labels" / "object_spatial_features.csv"

        summary = train_dimensions_baseline(
            csv_path=spatial_features_path,
            test_ratio=0.2,
            seed=args.seed,
        )

        print("Dimensions baseline summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")


    if args.vision_baseline_report:
        using_explicit_spatial_splits = (
            args.spatial_features_train is not None
            or args.spatial_features_val is not None
            or args.spatial_features_test is not None
        )

        if using_explicit_spatial_splits:
            if args.spatial_features_train is None or args.spatial_features_val is None or args.spatial_features_test is None:
                raise ValueError(
                    "--spatial-features-train, --spatial-features-val, and --spatial-features-test are all required for explicit split baseline reports"
                )

            output_path = args.vision_baseline_report_output
            if output_path is None:
                output_path = config.project_code / "datasets" / "vision_labels" / "vision_baseline_report_explicit_splits.json"

            output = run_vision_baseline_report_from_splits(
                train_csv=args.spatial_features_train,
                val_csv=args.spatial_features_val,
                test_csv=args.spatial_features_test,
                output_path=output_path,
                seed=args.seed,
            )
        else:
            spatial_features_path = args.spatial_features_input

            if spatial_features_path is None:
                spatial_features_path = config.project_code / "datasets" / "vision_labels" / "object_spatial_features.csv"

            output = run_vision_baseline_report(
                csv_path=spatial_features_path,
                output_path=args.vision_baseline_report_output,
                test_ratio=0.2,
                seed=args.seed,
            )

        print(f"Vision baseline report saved to: {output}")


    if args.export_vision_crops:
        vision_labels_path = args.vision_labels_input

        if vision_labels_path is None:
            vision_labels_path = config.project_code / "datasets" / "vision_labels" / "vision_labels_debug.jsonl"

        if args.unity_image_root is None:
            raise ValueError("--unity-image-root is required for --export-vision-crops")

        output_dir = args.vision_crops_output

        if output_dir is None:
            output_dir = config.project_code / "datasets" / "vision_crops" / "debug"

        output = export_vision_crops(
            vision_labels_path=vision_labels_path,
            unity_image_root=args.unity_image_root,
            output_dir=output_dir,
            padding=args.crop_padding,
            validate=True,
            include_depth=args.include_depth,
            include_target_mask=args.include_target_mask,
        )

        print(f"Vision crop labels saved to: {output}")

    if args.export_predicted_mask_crops:
        if args.crop_labels_input is None:
            raise ValueError("--crop-labels-input is required for --export-predicted-mask-crops")
        if args.weights is None:
            raise ValueError("--weights is required for --export-predicted-mask-crops")
        output_dir = args.predicted_mask_output
        if output_dir is None:
            output_dir = args.crop_labels_input.parent.with_name(
                f"{args.crop_labels_input.parent.name}_predicted_masks"
            )
        report = export_predicted_mask_crops(
            crop_labels_csv=args.crop_labels_input,
            weights=args.weights,
            output_dir=output_dir,
            classes=config.classes,
            confidence=args.conf,
            minimum_iou=args.predicted_mask_minimum_iou,
            image_size=args.imgsz or config.image_size,
            device=args.device,
            depth_mad_scale=args.predicted_mask_depth_mad_scale,
            minimum_depth_band_m=args.predicted_mask_depth_band_min,
            maximum_depth_band_m=args.predicted_mask_depth_band_max,
        )
        print("Predicted-mask crop export complete:")
        print(f"Matched rows: {report['num_matched_rows']}/{report['num_input_rows']}")
        print(f"Coverage: {report['coverage']:.4f}")
        print(f"Mean matched bbox IoU: {report['mean_matched_bbox_iou']:.4f}")
        print(f"Output: {report['crop_labels_output']}")

    if args.evaluate_depth_geometry:
        if args.crop_labels_input is None:
            raise ValueError("--crop-labels-input is required for --evaluate-depth-geometry")

        predictions_output = args.depth_geometry_predictions_output
        if predictions_output is None:
            predictions_output = args.crop_labels_input.parent / "depth_geometry_predictions.csv"

        report_output = args.depth_geometry_report_output
        if report_output is None:
            report_output = args.crop_labels_input.parent / "depth_geometry_report.json"

        report = evaluate_depth_geometry(
            crop_labels_csv=args.crop_labels_input,
            predictions_output=predictions_output,
            report_output=report_output,
            lower_quantile=args.geometry_lower_quantile,
            upper_quantile=args.geometry_upper_quantile,
        )

        print("Depth geometry evaluation complete:")
        print(f"Evaluated rows: {report['num_evaluated_rows']}")
        print(f"Skipped rows: {report['num_skipped_rows']}")
        print(f"Predictions: {predictions_output}")
        print(f"Report: {report_output}")
        print(f"Raw geometry: {report['raw_geometry']}")
        print(f"Robust geometry: {report['robust_geometry']}")

    if args.train_geometry_residual:
        if (
            args.geometry_features_train is None
            or args.geometry_features_val is None
            or args.geometry_features_test is None
        ):
            raise ValueError(
                "--geometry-features-train, --geometry-features-val, and "
                "--geometry-features-test are required for geometry residual training"
            )
        if args.geometry_residual_architecture is None:
            raise ValueError(
                "--geometry-residual-architecture is required for geometry residual training"
            )

        output_dir = args.geometry_residual_output
        if output_dir is None:
            output_dir = (
                config.yolo_runs_output
                / f"geometry_residual_{args.geometry_residual_architecture}"
            )

        summary = train_geometry_residual_from_splits(
            train_csv=args.geometry_features_train,
            val_csv=args.geometry_features_val,
            test_csv=args.geometry_features_test,
            output_dir=output_dir,
            architecture=args.geometry_residual_architecture,
            epochs=args.crop_spatial_epochs,
            batch_size=args.crop_spatial_batch,
            image_size=args.crop_spatial_image_size,
            minimum_depth_m=args.depth_min_m,
            maximum_depth_m=args.depth_max_m,
            seed=args.seed,
        )

        print("Geometry residual training complete:")
        print(f"Best checkpoint: {summary['best_model_path']}")
        print(f"Epochs completed: {summary['epochs_completed']}")
        print(f"Test metrics: {summary['test_metrics']}")

    if args.export_geometry_residual_predictions:
        if args.geometry_features_test is None:
            raise ValueError(
                "--geometry-features-test is required for geometry residual predictions"
            )
        if args.geometry_residual_checkpoint is None:
            raise ValueError(
                "--geometry-residual-checkpoint is required for geometry residual predictions"
            )

        predictions_output = args.geometry_residual_predictions_output
        if predictions_output is None:
            predictions_output = (
                args.geometry_residual_checkpoint.parent / "predictions.csv"
            )
        summary_output = args.geometry_residual_predictions_summary
        if summary_output is None:
            summary_output = predictions_output.with_name("prediction_summary.json")

        summary = export_geometry_residual_predictions(
            geometry_csv=args.geometry_features_test,
            checkpoint_path=args.geometry_residual_checkpoint,
            predictions_output=predictions_output,
            summary_output=summary_output,
            batch_size=args.crop_spatial_batch,
        )

        print(f"Geometry residual predictions saved to: {predictions_output}")
        print(f"Prediction summary saved to: {summary_output}")
        print(f"Prediction metrics: {summary['test_metrics']}")

    if args.train_crop_spatial_baseline:
        output_dir = args.crop_spatial_output

        if output_dir is None:
            output_dir = config.yolo_runs_output / "crop_spatial_baseline"

        using_explicit_splits = (
            args.crop_labels_train is not None
            or args.crop_labels_val is not None
            or args.crop_labels_test is not None
        )

        if using_explicit_splits:
            if args.crop_labels_train is None or args.crop_labels_val is None or args.crop_labels_test is None:
                raise ValueError(
                    "--crop-labels-train, --crop-labels-val, and --crop-labels-test are all required for explicit split training"
                )

            summary = train_crop_spatial_baseline_from_splits(
                train_csv=args.crop_labels_train,
                val_csv=args.crop_labels_val,
                test_csv=args.crop_labels_test,
                output_dir=output_dir,
                epochs=args.crop_spatial_epochs,
                batch_size=args.crop_spatial_batch,
                image_size=args.crop_spatial_image_size,
                seed=args.seed,
            )
        else:
            if args.crop_labels_input is None:
                raise ValueError("--crop-labels-input is required for --train-crop-spatial-baseline")

            summary = train_crop_spatial_baseline(
                crop_labels_csv=args.crop_labels_input,
                output_dir=output_dir,
                epochs=args.crop_spatial_epochs,
                batch_size=args.crop_spatial_batch,
                image_size=args.crop_spatial_image_size,
                test_ratio=0.2,
                seed=args.seed,
            )

        print("Crop spatial baseline summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")


    if args.train_multimodal_spatial:
        if args.crop_labels_train is None or args.crop_labels_val is None or args.crop_labels_test is None:
            raise ValueError(
                "--crop-labels-train, --crop-labels-val, and --crop-labels-test are required for multimodal training"
            )
        if args.spatial_architecture is None:
            raise ValueError("--spatial-architecture is required for multimodal training")

        output_dir = args.multimodal_spatial_output
        if output_dir is None:
            output_dir = config.yolo_runs_output / args.spatial_architecture

        summary = train_multimodal_spatial_from_splits(
            train_csv=args.crop_labels_train,
            val_csv=args.crop_labels_val,
            test_csv=args.crop_labels_test,
            output_dir=output_dir,
            architecture=args.spatial_architecture,
            epochs=args.crop_spatial_epochs,
            batch_size=args.crop_spatial_batch,
            image_size=args.crop_spatial_image_size,
            context_image_size=args.context_image_size,
            minimum_depth_m=args.depth_min_m,
            maximum_depth_m=args.depth_max_m,
            seed=args.seed,
        )

        print("Multimodal spatial model summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")


    if args.compare_vision_experiments:
        if args.bbox_baseline_report is None:
            raise ValueError("--bbox-baseline-report is required")

        if args.crop_baseline_summary is None:
            raise ValueError("--crop-baseline-summary is required")

        output = compare_vision_experiments(
            bbox_report_path=args.bbox_baseline_report,
            crop_summary_path=args.crop_baseline_summary,
            output_path=args.vision_comparison_output,
        )

        print(f"Vision experiment comparison saved to: {output}")


    if args.export_crop_spatial_predictions:
        if args.crop_labels_input is None:
            raise ValueError("--crop-labels-input is required for --export-crop-spatial-predictions")

        if args.crop_spatial_checkpoint is None:
            raise ValueError("--crop-spatial-checkpoint is required for --export-crop-spatial-predictions")

        output_csv = args.crop_spatial_predictions_output

        if output_csv is None:
            output_csv = args.crop_spatial_checkpoint.parent / "predictions.csv"

        output = export_crop_spatial_predictions(
            crop_labels_csv=args.crop_labels_input,
            checkpoint_path=args.crop_spatial_checkpoint,
            output_csv=output_csv,
        )

        summary_output = args.crop_spatial_predictions_summary

        if summary_output is None:
            summary_output = output.with_name("prediction_summary.json")

        summary = summarize_prediction_errors(
            prediction_csv=output,
            output_json=summary_output,
        )

        print(f"Crop spatial predictions saved to: {output}")
        print(f"Prediction summary saved to: {summary_output}")
        print("Prediction error summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")


    if args.export_multimodal_spatial_predictions:
        if args.crop_labels_input is None:
            raise ValueError(
                "--crop-labels-input is required for --export-multimodal-spatial-predictions"
            )
        if args.multimodal_spatial_checkpoint is None:
            raise ValueError(
                "--multimodal-spatial-checkpoint is required for --export-multimodal-spatial-predictions"
            )

        output_csv = args.multimodal_spatial_predictions_output
        if output_csv is None:
            output_csv = args.multimodal_spatial_checkpoint.parent / "predictions.csv"

        output = export_multimodal_spatial_predictions(
            crop_labels_csv=args.crop_labels_input,
            checkpoint_path=args.multimodal_spatial_checkpoint,
            output_csv=output_csv,
            batch_size=args.crop_spatial_batch,
        )

        summary_output = args.multimodal_spatial_predictions_summary
        if summary_output is None:
            summary_output = output.with_name("prediction_summary.json")

        summary = summarize_prediction_errors(
            prediction_csv=output,
            output_json=summary_output,
        )

        print(f"Multimodal spatial predictions saved to: {output}")
        print(f"Prediction summary saved to: {summary_output}")
        print("Prediction error summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")


    if not (
        args.yolo_conversion
        or args.export_yolo_segmentation
        or args.export_rgbd_yolo_segmentation
        or args.export_predicted_mask_crops
        or args.run_final_vision_pipeline
        or args.pipeline_ui
        or args.train_yolo
        or args.evaluate_yolo
        or args.predict_yolo
        or args.predict_rgbd_yolo
        or args.predict_depth_verified_yolo
        or args.export_detections
        or args.select_pose
        or args.export_vision_output
        or args.enrich_vision_semantic_labels
        or args.audit_vision_dataset
        or args.export_spatial_features
        or args.train_distance_baseline
        or args.train_position_baseline
        or args.train_dimensions_baseline
        or args.vision_baseline_report
        or args.export_vision_crops
        or args.evaluate_depth_geometry
        or args.train_geometry_residual
        or args.export_geometry_residual_predictions
        or args.train_crop_spatial_baseline
        or args.train_multimodal_spatial
        or args.compare_vision_experiments
        or args.export_crop_spatial_predictions
        or args.export_multimodal_spatial_predictions
    ):
        parser.print_help()


if __name__ == "__main__":
    main()
