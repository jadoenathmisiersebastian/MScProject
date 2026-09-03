from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from .vision_distance_baseline import (
    train_distance_baseline,
    train_distance_baseline_from_splits,
)
from .vision_position_baseline import (
    train_position_baseline,
    train_position_baseline_from_splits,
)
from .vision_dimensions_baseline import (
    train_dimensions_baseline,
    train_dimensions_baseline_from_splits,
)


def build_vision_baseline_report(
    csv_path: str | Path,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    return {
        "split_mode": "random_holdout",
        "csv_path": str(Path(csv_path).expanduser().resolve()),
        "test_ratio": test_ratio,
        "seed": seed,
        "distance_baseline": train_distance_baseline(
            csv_path=csv_path,
            test_ratio=test_ratio,
            seed=seed,
        ),
        "position_baseline": train_position_baseline(
            csv_path=csv_path,
            test_ratio=test_ratio,
            seed=seed,
        ),
        "dimensions_baseline": train_dimensions_baseline(
            csv_path=csv_path,
            test_ratio=test_ratio,
            seed=seed,
        ),
    }


def build_vision_baseline_report_from_splits(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    seed: int = 42,
) -> dict[str, Any]:
    return {
        "split_mode": "explicit_train_val_test",
        "train_csv": str(Path(train_csv).expanduser().resolve()),
        "val_csv": str(Path(val_csv).expanduser().resolve()),
        "test_csv": str(Path(test_csv).expanduser().resolve()),
        "seed": seed,
        "distance_baseline": train_distance_baseline_from_splits(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            seed=seed,
        ),
        "position_baseline": train_position_baseline_from_splits(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            seed=seed,
        ),
        "dimensions_baseline": train_dimensions_baseline_from_splits(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            seed=seed,
        ),
    }


def write_vision_baseline_report(
    report: dict[str, Any],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(report, f, indent=2)

    return output_path


def run_vision_baseline_report(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    csv_path = Path(csv_path).expanduser().resolve()

    if output_path is None:
        output_path = csv_path.with_name("vision_baseline_report.json")
    else:
        output_path = Path(output_path).expanduser().resolve()

    report = build_vision_baseline_report(
        csv_path=csv_path,
        test_ratio=test_ratio,
        seed=seed,
    )

    return write_vision_baseline_report(report, output_path)


def run_vision_baseline_report_from_splits(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    output_path: str | Path,
    seed: int = 42,
) -> Path:
    report = build_vision_baseline_report_from_splits(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        seed=seed,
    )
    return write_vision_baseline_report(report, output_path)
