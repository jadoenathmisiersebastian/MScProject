from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from src.final_vision_pipeline import (
    build_final_pipeline_stages,
    final_pipeline_config_fingerprint,
)
from src.final_vision_pipeline_ui import (
    inspect_raw_dataset,
    load_editable_pipeline_config,
    save_pipeline_ui_config,
    saved_state_requires_restart,
)


class FinalVisionPipelineTests(unittest.TestCase):
    def test_stage_builder_covers_complete_pipeline_and_uses_raw_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_splits = {
                split: str(root / "datasets" / "vision_raw" / f"final_{split}")
                for split in ("train", "val", "test")
            }
            config = {
                "pipeline_name": "final_test",
                "raw_splits": raw_splits,
                "device": "cpu",
            }

            stages = build_final_pipeline_stages(
                config=config,
                project_root=root,
                python_executable="python-test",
            )
            names = [stage.name for stage in stages]

            self.assertEqual(len(names), len(set(names)))
            self.assertIn("audit_train", names)
            self.assertIn("masked_crops_test", names)
            self.assertIn("yolo_segmentation_dataset", names)
            self.assertIn("train_yolo_segmentation", names)
            self.assertIn("evaluate_yolo_segmentation_test", names)
            self.assertIn("geometry_oracle_test", names)
            self.assertIn("geometry_predicted_masks_test", names)
            self.assertIn("geometry_predicted_bbox_test", names)
            self.assertIn("train_residual_predicted_masks", names)
            self.assertIn("three_way_comparison_figures", names)

            audit_train = next(stage for stage in stages if stage.name == "audit_train")
            self.assertEqual(audit_train.command[0], "python-test")
            self.assertIn(raw_splits["train"], audit_train.command)

            segmentation_export = next(
                stage for stage in stages
                if stage.name == "yolo_segmentation_dataset"
            )
            self.assertIn("--clear-split", segmentation_export.command)

    def test_pipeline_fingerprint_changes_with_dataset_paths(self):
        first = {
            "pipeline_name": "final_test",
            "raw_splits": {"train": "/a", "val": "/b", "test": "/c"},
            "device": "cpu",
            "config_path": "/ignored/config.json",
        }
        second = dict(first)
        second["raw_splits"] = {"train": "/new", "val": "/b", "test": "/c"}

        self.assertNotEqual(
            final_pipeline_config_fingerprint(first),
            final_pipeline_config_fingerprint(second),
        )
        first["config_path"] = "/another/ignored/config.json"
        self.assertEqual(
            final_pipeline_config_fingerprint(first),
            final_pipeline_config_fingerprint(
                {
                    **first,
                    "config_path": "/different/path.json",
                }
            ),
        )

    def test_ui_dataset_inspection_and_config_saving(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config" / "final.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps({
                "pipeline_name": "old_name",
                "raw_splits": {
                    "train": "old_train",
                    "val": "old_val",
                    "test": "old_test",
                },
                "device": "cpu",
                "segmentation_epochs": 7,
            }))

            raw_splits = {}
            for split, count in (("train", 3), ("val", 2), ("test", 1)):
                dataset = root / "raw" / split
                labels = dataset / "labels" / "vision_labels.jsonl"
                labels.parent.mkdir(parents=True)
                labels.write_text("{}\n" * count)
                raw_splits[split] = dataset

                readiness = inspect_raw_dataset(dataset)
                self.assertTrue(readiness.ready)
                self.assertEqual(readiness.frame_count, count)

            saved = save_pipeline_ui_config(
                config_path=config_path,
                project_root=root,
                pipeline_name="new_final_run",
                raw_splits=raw_splits,
            )
            _, reloaded = load_editable_pipeline_config(config_path, root)

            self.assertEqual(reloaded["pipeline_name"], "new_final_run")
            self.assertEqual(reloaded["segmentation_epochs"], 7)
            self.assertEqual(
                Path(reloaded["raw_splits"]["test"]),
                raw_splits["test"].resolve(),
            )
            self.assertIn("config_fingerprint", saved)

            state_path = root / "state.json"
            state_path.write_text(json.dumps({
                "config_fingerprint": saved["config_fingerprint"],
            }))
            self.assertFalse(saved_state_requires_restart(
                state_path,
                saved["config_fingerprint"],
            ))
            self.assertTrue(saved_state_requires_restart(state_path, "different"))

    def test_ui_dataset_inspection_rejects_missing_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            readiness = inspect_raw_dataset(directory)
            self.assertFalse(readiness.ready)
            self.assertIn("Missing", readiness.message)


if __name__ == "__main__":
    unittest.main()
