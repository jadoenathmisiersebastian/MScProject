from __future__ import annotations

from pathlib import Path
import csv
import json
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.yolo_segmentation_dataset import (
    export_yolo_segmentation_split,
    mask_to_yolo_polygon,
)
from src.vision_predicted_masks import (
    bbox_iou,
    filter_predicted_mask_depth,
    rasterize_bbox_crop,
    rasterize_polygon_crop,
    select_prediction_index,
)
from src.vision_segmentation_figures import binary_mask_metrics


class YoloSegmentationDatasetTests(unittest.TestCase):
    def test_binary_mask_metrics(self):
        truth = np.asarray([[1, 1], [0, 0]], dtype=bool)
        prediction = np.asarray([[1, 0], [1, 0]], dtype=bool)

        metrics = binary_mask_metrics(truth, prediction)

        self.assertAlmostEqual(metrics["iou"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["dice"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)

    def test_predicted_bbox_is_rasterized_in_crop_coordinates(self):
        mask = rasterize_bbox_crop(
            bbox_xyxy=np.asarray([12.2, 22.2, 16.8, 25.8]),
            crop_box=(10, 20, 20, 28),
        )

        self.assertEqual(mask.shape, (8, 10))
        self.assertTrue(mask[2, 2])
        self.assertTrue(mask[5, 6])
        self.assertFalse(mask[0, 0])

    def test_depth_support_filter_rejects_background_leak(self):
        depth = np.ones((6, 6), dtype=np.float32)
        depth[:, 5] = 2.0
        predicted_mask = np.ones((6, 6), dtype=bool)

        support, metadata = filter_predicted_mask_depth(
            depth=depth,
            predicted_mask=predicted_mask,
        )

        self.assertTrue(np.all(support[:, :5]))
        self.assertFalse(np.any(support[:, 5]))
        self.assertAlmostEqual(metadata["depth_filter_center_m"], 1.0)
        self.assertAlmostEqual(metadata["depth_filter_band_m"], 0.08)
        self.assertAlmostEqual(metadata["depth_filter_retained_fraction"], 5.0 / 6.0)

    def test_prediction_matching_filters_class_and_uses_best_iou(self):
        boxes = np.asarray([
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 9.0, 9.0],
            [2.0, 2.0, 8.0, 8.0],
        ])
        classes = np.asarray([1, 1, 2])
        index, overlap = select_prediction_index(
            ground_truth_bbox=np.asarray([1.0, 1.0, 9.0, 9.0]),
            ground_truth_class_id=1,
            predicted_boxes=boxes,
            predicted_class_ids=classes,
        )

        self.assertEqual(index, 1)
        self.assertAlmostEqual(overlap, 1.0)
        self.assertAlmostEqual(bbox_iou(boxes[0], boxes[1]), 0.64)

        index, _ = select_prediction_index(
            ground_truth_bbox=np.asarray([1.0, 1.0, 9.0, 9.0]),
            ground_truth_class_id=1,
            predicted_boxes=boxes,
            predicted_class_ids=classes,
            used_indices={1},
        )
        self.assertEqual(index, 0)

    def test_predicted_polygon_is_rasterized_in_crop_coordinates(self):
        polygon = np.asarray([
            [12.0, 22.0],
            [16.0, 22.0],
            [16.0, 25.0],
            [12.0, 25.0],
        ])
        mask = rasterize_polygon_crop(polygon, (10, 20, 20, 28))

        self.assertEqual(mask.shape, (8, 10))
        self.assertTrue(mask[2, 2])
        self.assertTrue(mask[5, 6])
        self.assertFalse(mask[0, 0])

    def test_mask_to_yolo_polygon_uses_largest_component_and_source_coordinates(self):
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[1:5, 2:7] = 1
        mask[6:8, 8:10] = 1

        polygon, component_count, retained_fraction = mask_to_yolo_polygon(
            mask=mask,
            crop_left=10,
            crop_top=20,
            image_width=100,
            image_height=100,
            simplification_tolerance=0.0,
        )

        points = np.asarray(polygon).reshape(-1, 2)
        self.assertEqual(component_count, 2)
        self.assertGreater(retained_fraction, 0.75)
        self.assertTrue(np.all(points[:, 0] >= 0.12))
        self.assertTrue(np.all(points[:, 0] <= 0.16))
        self.assertTrue(np.all(points[:, 1] >= 0.21))
        self.assertTrue(np.all(points[:, 1] <= 0.24))

    def test_export_segmentation_split_writes_yolo_dataset_and_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_image = root / "source.png"
            target_mask = root / "mask.png"
            crop_labels = root / "crop_labels.csv"
            output = root / "yolo_seg"

            Image.new("RGB", (20, 12), (80, 120, 160)).save(source_image)
            mask = np.zeros((8, 10), dtype=np.uint8)
            mask[1:7, 2:9] = 255
            Image.fromarray(mask, mode="L").save(target_mask)

            row = {
                "source_image_path": str(source_image),
                "target_mask_path": str(target_mask),
                "semantic_class": "bottle",
                "frame_id": "frame_000001",
                "crop_left": "4",
                "crop_top": "2",
                "source_image_width": "20",
                "source_image_height": "12",
            }
            with crop_labels.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

            null_image = root / "null.png"
            Image.new("RGB", (20, 12), (20, 30, 40)).save(null_image)
            manifest_row = {
                "frame_id": "frame_000002",
                "scene_id": "negative_test",
                "source_image_path": str(null_image),
                "is_null_sample": "1",
                "num_target_objects": "0",
            }
            with (root / "frame_manifest.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(manifest_row))
                writer.writeheader()
                writer.writerow(manifest_row)

            summary = export_yolo_segmentation_split(
                crop_labels_csv=crop_labels,
                output_path=output,
                split="train",
                classes={"bottle": 0},
                clear_split=True,
            )

            images = list((output / "images" / "train").glob("*.png"))
            labels = list((output / "labels" / "train").glob("*.txt"))
            self.assertEqual(len(images), 2)
            self.assertEqual(len(labels), 2)
            self.assertEqual(summary.split_counts, {"train": 2})
            self.assertEqual(summary.object_counts, {"train": 1})
            self.assertEqual(summary.negative_frame_count, 1)
            self.assertEqual(summary.skipped_objects, 0)

            positive_label = next(label for label in labels if label.read_text().strip())
            negative_label = next(label for label in labels if not label.read_text().strip())
            values = positive_label.read_text().split()
            self.assertEqual(values[0], "0")
            self.assertGreaterEqual(len(values), 7)
            coordinates = [float(value) for value in values[1:]]
            self.assertTrue(all(0.0 <= value <= 1.0 for value in coordinates))
            self.assertEqual(negative_label.read_text(), "")

            yaml_text = (output / "data.yaml").read_text()
            self.assertIn("train: images/train", yaml_text)
            self.assertIn("0: bottle", yaml_text)

            report = json.loads((output / "export_summary_train.json").read_text())
            self.assertEqual(report["split_counts"], {"train": 2})
            self.assertEqual(report["object_counts"], {"train": 1})
            self.assertEqual(report["negative_frame_count"], 1)


if __name__ == "__main__":
    unittest.main()
