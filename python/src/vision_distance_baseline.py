from __future__ import annotations

from pathlib import Path
import csv
import math
import random
from typing import Any


FEATURE_COLUMNS = [
    "bbox_width",
    "bbox_height",
    "bbox_center_x_norm",
    "bbox_center_y_norm",
    "bbox_area_normalized",
]

TARGET_COLUMN = "distance_camera_m"
CLASS_COLUMN = "semantic_class"


def _read_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(csv_path).expanduser().resolve()

    with csv_path.open("r") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _class_mapping(rows: list[dict[str, Any]]) -> dict[str, int]:
    classes = sorted({row[CLASS_COLUMN] for row in rows})
    return {name: index for index, name in enumerate(classes)}


def _features(row: dict[str, Any], class_to_id: dict[str, int]) -> list[float]:
    values = [_float(row, key) for key in FEATURE_COLUMNS]
    values.append(float(class_to_id[row[CLASS_COLUMN]]))
    return values


def _target(row: dict[str, Any]) -> float:
    return _float(row, TARGET_COLUMN)


def _train_test_split(rows: list[dict[str, Any]], test_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    test_count = max(1, int(len(shuffled) * test_ratio))
    return shuffled[test_count:], shuffled[:test_count]


def _evaluate(model, rows: list[dict[str, Any]], class_to_id: dict[str, int]) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    x = [_features(row, class_to_id) for row in rows]
    y = [_target(row) for row in rows]
    predictions = model.predict(x)

    mse = mean_squared_error(y, predictions)
    return {
        "num_rows": len(rows),
        "mae_m": mean_absolute_error(y, predictions),
        "rmse_m": math.sqrt(mse),
        "r2": r2_score(y, predictions) if len(y) > 1 else float("nan"),
    }


def _fit_model(train_rows: list[dict[str, Any]], class_to_id: dict[str, int], seed: int):
    from sklearn.ensemble import RandomForestRegressor

    x_train = [_features(row, class_to_id) for row in train_rows]
    y_train = [_target(row) for row in train_rows]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def train_distance_baseline(
    csv_path: str | Path,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    rows = _read_rows(csv_path)

    if len(rows) < 5:
        raise ValueError("Need at least 5 rows for a baseline train/test split.")

    class_to_id = _class_mapping(rows)
    train_rows, test_rows = _train_test_split(rows, test_ratio=test_ratio, seed=seed)
    model = _fit_model(train_rows, class_to_id, seed)
    test_metrics = _evaluate(model, test_rows, class_to_id)

    return {
        "split_mode": "random_holdout",
        "num_rows": len(rows),
        "num_train": len(train_rows),
        "num_test": len(test_rows),
        "class_to_id": class_to_id,
        "feature_columns": FEATURE_COLUMNS + ["class_id"],
        "target_column": TARGET_COLUMN,
        **test_metrics,
    }


def train_distance_baseline_from_splits(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    seed: int = 42,
) -> dict[str, Any]:
    train_rows = _read_rows(train_csv)
    val_rows = _read_rows(val_csv)
    test_rows = _read_rows(test_csv)

    if len(train_rows) < 1 or len(val_rows) < 1 or len(test_rows) < 1:
        raise ValueError("Train, validation, and test CSVs must each contain at least one row.")

    class_to_id = _class_mapping(train_rows + val_rows + test_rows)
    model = _fit_model(train_rows, class_to_id, seed)

    return {
        "split_mode": "explicit_train_val_test",
        "train_csv": str(Path(train_csv).expanduser().resolve()),
        "val_csv": str(Path(val_csv).expanduser().resolve()),
        "test_csv": str(Path(test_csv).expanduser().resolve()),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_test": len(test_rows),
        "class_to_id": class_to_id,
        "feature_columns": FEATURE_COLUMNS + ["class_id"],
        "target_column": TARGET_COLUMN,
        "val_metrics": _evaluate(model, val_rows, class_to_id),
        "test_metrics": _evaluate(model, test_rows, class_to_id),
    }
