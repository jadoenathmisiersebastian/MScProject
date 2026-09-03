from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from src.vision_crop_dataset import _depth_annotation, export_vision_crops
from src.vision_depth_geometry import (
    backproject_masked_depth,
    evaluate_depth_geometry,
    estimate_depth_geometry,
)
from src.vision_crop_spatial_baseline import (
    METADATA_COLUMNS,
    TARGET_COLUMNS,
    MetadataNormalizer,
    TargetNormalizer,
)
from src.vision_multimodal_spatial import (
    DEPTH_ONLY,
    RGBD_DUAL_CONTEXT,
    RGB_DUAL_CONTEXT,
    MultimodalSpatialDataset,
    MultimodalSpatialRegressor,
    ResizeWithPadding,
    _load_depth_tensor,
)


class VisionMultimodalSpatialTests(unittest.TestCase):
    def test_resize_with_padding_preserves_canvas_size(self):
        image = Image.new("RGB", (120, 40), (255, 0, 0))
        resized = ResizeWithPadding(64)(image)

        self.assertEqual(resized.size, (64, 64))
        self.assertEqual(resized.getpixel((32, 32)), (255, 0, 0))
        self.assertNotEqual(resized.getpixel((32, 0)), (255, 0, 0))

    def test_depth_tensor_preserves_metric_scale_and_validity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.npy"
            np.save(
                path,
                np.array([[0.0, 0.3], [1.65, 3.0]], dtype=np.float32),
            )

            depth = _load_depth_tensor(
                path=path,
                image_size=2,
                minimum_depth_m=0.3,
                maximum_depth_m=3.0,
            )

            self.assertEqual(tuple(depth.shape), (2, 2, 2))
            self.assertAlmostEqual(float(depth[0, 0, 0]), 0.0)
            self.assertAlmostEqual(float(depth[0, 1, 0]), 0.5, places=5)
            self.assertAlmostEqual(float(depth[0, 1, 1]), 1.0, places=5)
            self.assertEqual(depth[1].tolist(), [[0.0, 1.0], [1.0, 1.0]])

    def test_depth_annotation_resolves_solo_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "solo" / "sequence.0"
            sequence.mkdir(parents=True)
            depth_path = sequence / "step0.camera.depth.exr"
            depth_path.touch()
            frame_data = {
                "captures": [{
                    "annotations": [{
                        "@type": "type.unity.com/unity.solo.DepthAnnotation",
                        "filename": depth_path.name,
                        "measurementStrategy": "Depth",
                    }],
                }],
            }
            (sequence / "step0.frame_data.json").write_text(json.dumps(frame_data))

            resolved, strategy = _depth_annotation(
                root,
                "solo/sequence.0/step0.camera.png",
            )

            self.assertEqual(resolved, depth_path)
            self.assertEqual(strategy, "Depth")

    def test_export_vision_crops_writes_aligned_metric_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "raw" / "solo" / "sequence.0"
            labels = root / "raw" / "labels"
            output = root / "crops"
            sequence.mkdir(parents=True)
            labels.mkdir(parents=True)

            rgb_path = sequence / "step0.camera.png"
            depth_path = sequence / "step0.camera.depth.exr"
            Image.new("RGB", (6, 4), (80, 120, 160)).save(rgb_path)
            depth_path.touch()

            frame_data = {
                "captures": [{
                    "annotations": [{
                        "@type": "type.unity.com/unity.solo.DepthAnnotation",
                        "filename": depth_path.name,
                        "measurementStrategy": "Depth",
                    }],
                }],
            }
            (sequence / "step0.frame_data.json").write_text(json.dumps(frame_data))

            record = {
                "frame_id": "frame_000000",
                "scene_id": "depth_test",
                "image_path": "solo/sequence.0/step0.camera.png",
                "objects": [{
                    "object_id": 1,
                    "object_name": "Bottle",
                    "class_name": "Bottle",
                    "semantic_class": "bottle",
                    "is_focused_object": True,
                    "bbox_xyxy": [1.0, 1.0, 5.0, 3.0],
                    "bbox_area_normalized": 1.0 / 3.0,
                    "distance_camera_m": 1.2,
                    "position_camera": [0.1, -0.2, 1.2],
                    "dimensions_m": [0.08, 0.22, 0.08],
                }],
            }
            label_path = labels / "vision_labels_semantic.jsonl"
            label_path.write_text(json.dumps(record) + "\n")
            metric_depth = np.arange(24, dtype=np.float32).reshape(4, 6) + 1.0

            with patch(
                "src.vision_crop_dataset._read_metric_depth",
                return_value=metric_depth,
            ):
                csv_path = export_vision_crops(
                    vision_labels_path=label_path,
                    unity_image_root=root / "raw",
                    output_dir=output,
                    padding=0.0,
                    validate=False,
                    include_depth=True,
                )

            with csv_path.open(newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            depth_crop = np.load(rows[0]["depth_crop_path"])
            np.testing.assert_array_equal(depth_crop, metric_depth[1:3, 1:5])
            self.assertEqual(rows[0]["depth_measurement_strategy"], "Depth")
            self.assertAlmostEqual(float(rows[0]["depth_valid_fraction"]), 1.0)

    def test_export_vision_crops_writes_target_mask_and_masked_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "raw" / "solo" / "sequence.0"
            labels = root / "raw" / "labels"
            output = root / "crops"
            sequence.mkdir(parents=True)
            labels.mkdir(parents=True)

            rgb_path = sequence / "step0.camera.png"
            depth_path = sequence / "step0.camera.depth.exr"
            segmentation_path = sequence / "step0.camera.semantic.png"
            Image.new("RGB", (6, 4), (80, 120, 160)).save(rgb_path)
            depth_path.touch()

            segmentation = np.zeros((4, 6, 4), dtype=np.uint8)
            segmentation[:, :, 3] = 255
            segmentation[1:3, 2:4] = np.array([10, 20, 30, 255])
            Image.fromarray(segmentation, mode="RGBA").save(segmentation_path)

            frame_data = {
                "captures": [{
                    "filename": rgb_path.name,
                    "dimension": [6, 4],
                    "matrix": [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -1.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "annotations": [
                        {
                            "@type": "type.unity.com/unity.solo.DepthAnnotation",
                            "filename": depth_path.name,
                            "measurementStrategy": "Depth",
                        },
                        {
                            "@type": "type.unity.com/unity.solo.SemanticSegmentationAnnotation",
                            "filename": segmentation_path.name,
                            "instances": [{
                                "labelName": "bottle",
                                "pixelValue": [10, 20, 30, 255],
                            }],
                        },
                    ],
                }],
            }
            (sequence / "step0.frame_data.json").write_text(json.dumps(frame_data))

            record = {
                "frame_id": "frame_000000",
                "scene_id": "masked_depth_test",
                "image_path": "solo/sequence.0/step0.camera.png",
                "objects": [{
                    "object_id": 1,
                    "object_name": "Bottle",
                    "class_name": "Bottle",
                    "semantic_class": "bottle",
                    "is_focused_object": True,
                    "bbox_xyxy": [1.0, 1.0, 5.0, 3.0],
                    "bbox_area_normalized": 1.0 / 3.0,
                    "distance_camera_m": 1.2,
                    "position_camera": [0.1, -0.2, 1.2],
                    "dimensions_m": [0.08, 0.22, 0.08],
                }],
            }
            label_path = labels / "vision_labels_semantic.jsonl"
            label_path.write_text(json.dumps(record) + "\n")
            metric_depth = np.arange(24, dtype=np.float32).reshape(4, 6) + 1.0

            with patch(
                "src.vision_crop_dataset._read_metric_depth",
                return_value=metric_depth,
            ):
                csv_path = export_vision_crops(
                    vision_labels_path=label_path,
                    unity_image_root=root / "raw",
                    output_dir=output,
                    padding=0.0,
                    validate=False,
                    include_depth=True,
                    include_target_mask=True,
                )

            with csv_path.open(newline="") as file:
                row = next(csv.DictReader(file))

            mask = np.asarray(Image.open(row["target_mask_path"])) > 0
            masked_depth = np.load(row["masked_depth_crop_path"])
            expected_mask = np.array([
                [False, True, True, False],
                [False, True, True, False],
            ])

            np.testing.assert_array_equal(mask, expected_mask)
            np.testing.assert_array_equal(
                masked_depth,
                metric_depth[1:3, 1:5] * expected_mask,
            )
            self.assertAlmostEqual(float(row["target_mask_fraction"]), 0.5)
            self.assertAlmostEqual(float(row["target_depth_valid_fraction"]), 1.0)
            self.assertAlmostEqual(float(row["camera_fx_px"]), 6.0)
            self.assertAlmostEqual(float(row["camera_fy_px"]), 4.0)

    def test_depth_geometry_backprojection_and_report(self):
        depth = np.zeros((3, 3), dtype=np.float32)
        depth[1, 1] = 1.0
        points = backproject_masked_depth(
            masked_depth=depth,
            crop_left=0.0,
            crop_top=0.0,
            fx=100.0,
            fy=100.0,
            cx=1.5,
            cy=1.5,
            measurement_strategy="Depth",
        )
        np.testing.assert_allclose(points, [[0.0, 0.0, 1.0]], atol=1e-7)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            depth_path = root / "masked.npy"
            labels_path = root / "crop_labels.csv"
            predictions_path = root / "predictions.csv"
            report_path = root / "report.json"
            np.save(depth_path, depth)

            row = {
                "frame_id": "frame_000000",
                "scene_id": "geometry_test",
                "object_name": "Bottle",
                "semantic_class": "bottle",
                "masked_depth_crop_path": str(depth_path),
                "target_mask_fraction": 1.0 / 9.0,
                "target_depth_valid_fraction": 1.0,
                "depth_measurement_strategy": "Depth",
                "depth_crop_left": 0,
                "depth_crop_top": 0,
                "camera_fx_px": 100.0,
                "camera_fy_px": 100.0,
                "camera_cx_px": 1.5,
                "camera_cy_px": 1.5,
                "camera_rotation_world_x": 0.0,
                "camera_rotation_world_y": 0.0,
                "camera_rotation_world_z": 0.0,
                "camera_rotation_world_w": 1.0,
            }
            estimate = estimate_depth_geometry(row)
            row.update({
                "distance_camera_m": estimate["raw_distance_camera_m"],
                "position_camera_x": estimate["raw_center_camera"][0],
                "position_camera_y": estimate["raw_center_camera"][1],
                "position_camera_z": estimate["raw_center_camera"][2],
                "dimension_x": estimate["raw_dimensions_world"][0],
                "dimension_y": estimate["raw_dimensions_world"][1],
                "dimension_z": estimate["raw_dimensions_world"][2],
            })

            with labels_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)

            report = evaluate_depth_geometry(
                crop_labels_csv=labels_path,
                predictions_output=predictions_path,
                report_output=report_path,
            )

            self.assertEqual(report["num_evaluated_rows"], 1)
            self.assertEqual(report["num_skipped_rows"], 0)
            self.assertAlmostEqual(
                report["raw_geometry"]["mae_distance_camera_m"],
                0.0,
            )
            self.assertTrue(predictions_path.exists())
            self.assertTrue(report_path.exists())

    def test_all_architectures_produce_seven_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crop_path = root / "crop.png"
            context_path = root / "context.png"
            depth_path = root / "depth.npy"
            Image.new("RGB", (40, 60), (80, 120, 160)).save(crop_path)
            Image.new("RGB", (120, 70), (100, 130, 170)).save(context_path)
            np.save(depth_path, np.full((40, 60), 1.2, dtype=np.float32))

            base_row = {
                **{column: str(index + 1) for index, column in enumerate(METADATA_COLUMNS)},
                **{column: str(0.1 * (index + 1)) for index, column in enumerate(TARGET_COLUMNS)},
                "crop_image_path": str(crop_path),
                "source_image_path": str(context_path),
                "depth_crop_path": str(depth_path),
                "semantic_class": "bottle",
            }
            second_row = {
                **{column: str((index + 1) * 2) for index, column in enumerate(METADATA_COLUMNS)},
                **{column: str(0.2 * (index + 1)) for index, column in enumerate(TARGET_COLUMNS)},
                "crop_image_path": str(crop_path),
                "source_image_path": str(context_path),
                "depth_crop_path": str(depth_path),
                "semantic_class": "glass",
            }
            rows = [base_row, second_row]
            class_to_id = {"bottle": 0, "glass": 1}
            metadata_normalizer = MetadataNormalizer.from_rows(rows, METADATA_COLUMNS)
            target_normalizer = TargetNormalizer.from_targets([
                [float(row[column]) for column in TARGET_COLUMNS]
                for row in rows
            ])

            for architecture in (
                RGB_DUAL_CONTEXT,
                DEPTH_ONLY,
                RGBD_DUAL_CONTEXT,
            ):
                with self.subTest(architecture=architecture):
                    dataset = MultimodalSpatialDataset(
                        rows=rows,
                        architecture=architecture,
                        class_to_id=class_to_id,
                        target_normalizer=target_normalizer,
                        metadata_normalizer=metadata_normalizer,
                        image_size=64,
                        context_image_size=64,
                        minimum_depth_m=0.3,
                        maximum_depth_m=3.0,
                    )
                    sample = dataset[0]
                    model = MultimodalSpatialRegressor(
                        architecture=architecture,
                        metadata_dim=len(METADATA_COLUMNS) + len(class_to_id),
                        output_dim=len(TARGET_COLUMNS),
                        use_pretrained=False,
                    )
                    model.eval()
                    with torch.no_grad():
                        output = model(
                            metadata=sample["metadata"].unsqueeze(0),
                            rgb_crop=sample.get("rgb_crop", None).unsqueeze(0)
                            if "rgb_crop" in sample else None,
                            rgb_context=sample.get("rgb_context", None).unsqueeze(0)
                            if "rgb_context" in sample else None,
                            depth_crop=sample.get("depth_crop", None).unsqueeze(0)
                            if "depth_crop" in sample else None,
                        )

                    self.assertEqual(tuple(output.shape), (1, len(TARGET_COLUMNS)))


if __name__ == "__main__":
    unittest.main()
