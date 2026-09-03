from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch

from src.vision_crop_spatial_baseline import (
    LEGACY_METADATA_COLUMNS,
    METADATA_COLUMNS,
    MetadataNormalizer,
    build_metadata_tensor,
    prepare_crop_rows,
)


class VisionCropSpatialV2Tests(unittest.TestCase):
    def test_prepare_crop_rows_derives_normalized_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "source.png"
            Image.new("RGB", (200, 100)).save(image_path)

            row = {
                "source_image_path": str(image_path),
                "bbox_x1": "20",
                "bbox_y1": "10",
                "bbox_x2": "120",
                "bbox_y2": "60",
                "bbox_area_normalized": "0.25",
            }

            prepare_crop_rows([row])

            self.assertEqual(row["source_image_width"], "200")
            self.assertEqual(row["source_image_height"], "100")
            self.assertAlmostEqual(float(row["bbox_width_norm"]), 0.5)
            self.assertAlmostEqual(float(row["bbox_height_norm"]), 0.5)
            self.assertAlmostEqual(float(row["bbox_center_x_norm"]), 0.35)
            self.assertAlmostEqual(float(row["bbox_center_y_norm"]), 0.35)
            self.assertAlmostEqual(float(row["bbox_aspect_ratio"]), 2.0)

    def test_v2_metadata_is_standardized_and_class_is_one_hot(self):
        rows = [
            {
                **{column: str(index + 1) for index, column in enumerate(METADATA_COLUMNS)},
                "semantic_class": "bottle",
            },
            {
                **{column: str((index + 1) * 2) for index, column in enumerate(METADATA_COLUMNS)},
                "semantic_class": "glass",
            },
        ]
        class_to_id = {"bottle": 0, "glass": 1}
        normalizer = MetadataNormalizer.from_rows(rows, METADATA_COLUMNS)

        metadata = build_metadata_tensor(
            row=rows[0],
            class_to_id=class_to_id,
            metadata_columns=METADATA_COLUMNS,
            class_encoding="one_hot",
            metadata_normalizer=normalizer,
        )

        self.assertEqual(tuple(metadata.shape), (len(METADATA_COLUMNS) + 2,))
        self.assertTrue(torch.isfinite(metadata).all())
        self.assertEqual(metadata[-2:].tolist(), [1.0, 0.0])

    def test_legacy_metadata_keeps_ordinal_class_encoding(self):
        row = {
            **{column: "1" for column in LEGACY_METADATA_COLUMNS},
            "class_name": "glass_prefab",
        }

        metadata = build_metadata_tensor(
            row=row,
            class_to_id={"bottle_prefab": 0, "glass_prefab": 1},
            metadata_columns=LEGACY_METADATA_COLUMNS,
            class_encoding="ordinal",
            class_column="class_name",
        )

        self.assertEqual(tuple(metadata.shape), (len(LEGACY_METADATA_COLUMNS) + 1,))
        self.assertEqual(float(metadata[-1]), 1.0)


if __name__ == "__main__":
    unittest.main()
