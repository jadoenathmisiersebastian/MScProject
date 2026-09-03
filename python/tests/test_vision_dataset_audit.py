from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.vision_dataset_audit import audit_and_filter_vision_dataset


def _vision_record(frame_index: int) -> dict:
    return {
        "frame_id": f"frame_{frame_index:06d}",
        "scene_id": "audit_test",
        "image_path": f"solo/sequence.0/step{frame_index}.camera.png",
        "camera": {
            "position_world": [0.0, 0.0, 0.0],
            "rotation_world_quat": [0.0, 0.0, 0.0, 1.0],
            "image_width": 4,
            "image_height": 4,
            "field_of_view_degrees": 50.0,
        },
        "objects": [{
            "object_id": frame_index + 1,
            "object_name": "Bottle",
            "class_name": "Bottle",
            "semantic_class": "bottle",
            "bbox_xyxy": [1.0, 1.0, 3.0, 3.0],
            "image_center": [2.0, 2.0],
            "normalized_center": [0.5, 0.5],
            "bbox_area_pixels": 4.0,
            "bbox_area_normalized": 0.25,
            "position_camera": [0.0, 0.0, 1.0],
            "distance_camera_m": 1.0,
            "dimensions_m": [0.1, 0.2, 0.1],
            "is_in_front_of_camera": True,
            "focus_distance": 0.0,
            "is_focused_object": True,
        }],
    }


def _frame_data(frame_index: int, visible: bool) -> dict:
    bbox_values = [{
        "origin": [1.0, 1.0],
        "dimension": [2.0, 2.0],
        "labelName": "bottle",
    }] if visible else []
    rendered_values = [{"labels": ["bottle"]}] if visible else []

    return {
        "captures": [{
            "annotations": [
                {
                    "@type": "type.unity.com/unity.solo.BoundingBox2DAnnotation",
                    "values": bbox_values,
                },
                {
                    "@type": "type.unity.com/unity.solo.SemanticSegmentationAnnotation",
                    "filename": f"step{frame_index}.camera.semantic segmentation.png",
                },
                {
                    "@type": "type.unity.com/unity.solo.DepthAnnotation",
                    "filename": f"step{frame_index}.camera.Depth.exr",
                },
            ],
        }],
        "metrics": [{
            "@type": "type.unity.com/unity.solo.RenderedObjectInfoMetric",
            "values": rendered_values,
        }],
    }


class VisionDatasetAuditTests(unittest.TestCase):
    def test_audit_accepts_explicit_null_frame_with_empty_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "solo" / "sequence.0"
            labels = root / "labels"
            sequence.mkdir(parents=True)
            labels.mkdir(parents=True)

            record = _vision_record(0)
            record["objects"] = []
            record["is_null_sample"] = True
            raw_path = labels / "vision_labels.jsonl"
            raw_path.write_text(json.dumps(record) + "\n")

            Image.new("RGB", (4, 4), (100, 120, 140)).save(
                sequence / "step0.camera.png"
            )
            Image.new("RGB", (4, 4), (0, 0, 0)).save(
                sequence / "step0.camera.semantic segmentation.png"
            )
            (sequence / "step0.camera.Depth.exr").touch()
            (sequence / "step0.frame_data.json").write_text(
                json.dumps(_frame_data(0, False))
            )

            filtered_path = labels / "vision_labels_valid.jsonl"
            report = audit_and_filter_vision_dataset(
                vision_labels_path=raw_path,
                dataset_root=root,
                filtered_output_path=filtered_path,
                report_output_path=labels / "audit.json",
                require_depth=True,
            )

            self.assertEqual(report["num_valid_frames"], 1)
            self.assertEqual(report["num_valid_null_frames"], 1)
            self.assertEqual(report["num_valid_objects"], 0)
            self.assertEqual(report["issue_counts"], {})

    def test_audit_filters_fully_occluded_frame_and_preserves_raw_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "solo" / "sequence.0"
            labels = root / "labels"
            sequence.mkdir(parents=True)
            labels.mkdir(parents=True)

            records = [_vision_record(0), _vision_record(1)]
            raw_path = labels / "vision_labels.jsonl"
            raw_contents = "".join(
                json.dumps(record) + "\n"
                for record in records
            )
            raw_path.write_text(raw_contents)

            for frame_index, visible in ((0, True), (1, False)):
                Image.new("RGB", (4, 4), (100, 120, 140)).save(
                    sequence / f"step{frame_index}.camera.png"
                )
                segmentation = Image.new("RGB", (4, 4), (0, 0, 0))
                if visible:
                    for x in range(1, 3):
                        for y in range(1, 3):
                            segmentation.putpixel((x, y), (0, 255, 0))
                segmentation.save(
                    sequence / f"step{frame_index}.camera.semantic segmentation.png"
                )
                (sequence / f"step{frame_index}.camera.Depth.exr").touch()
                (sequence / f"step{frame_index}.frame_data.json").write_text(
                    json.dumps(_frame_data(frame_index, visible))
                )

            filtered_path = labels / "vision_labels_valid.jsonl"
            report_path = labels / "vision_labels_valid.audit.json"
            report = audit_and_filter_vision_dataset(
                vision_labels_path=raw_path,
                dataset_root=root,
                filtered_output_path=filtered_path,
                report_output_path=report_path,
                require_depth=True,
            )

            self.assertEqual(raw_path.read_text(), raw_contents)
            self.assertEqual(report["num_input_frames"], 2)
            self.assertEqual(report["num_valid_frames"], 1)
            self.assertEqual(report["num_invalid_frames"], 1)
            self.assertEqual(report["invalid_frames"][0]["frame_id"], "frame_000001")
            self.assertEqual(report["issue_counts"]["bbox_count_mismatch"], 1)
            self.assertEqual(report["issue_counts"]["insufficient_visible_pixels"], 1)
            self.assertEqual(report["issue_counts"]["rendered_object_count_mismatch"], 1)
            self.assertTrue(report_path.exists())

            filtered_records = [
                json.loads(line)
                for line in filtered_path.read_text().splitlines()
            ]
            self.assertEqual(
                [record["frame_id"] for record in filtered_records],
                ["frame_000000"],
            )


if __name__ == "__main__":
    unittest.main()
