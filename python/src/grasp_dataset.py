from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
from typing import Any, Iterable

from .grasp_trial_schema import validate_grasp_trial


SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".ndjson"}


def _records_from_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "trials" in payload:
        trials = payload["trials"]
        if not isinstance(trials, list):
            raise ValueError("JSON field 'trials' must be a list")
        return trials
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("JSON grasp dataset must be a record, a list of records, or an object with a 'trials' list")


def load_grasp_trial_file(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Grasp trial file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported grasp trial extension: {path.suffix}")

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = []
        with path.open("r") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
        return records

    with path.open("r") as f:
        return _records_from_json_payload(json.load(f))


def iter_grasp_trial_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for item in paths:
        path = Path(item).expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(child for child in path.rglob("*") if child.suffix.lower() in SUPPORTED_EXTENSIONS))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Grasp trial path does not exist: {path}")
    return files


def load_grasp_trials(paths: Iterable[str | Path], validate: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_grasp_trial_files(paths):
        for index, record in enumerate(load_grasp_trial_file(path)):
            if validate:
                result = validate_grasp_trial(record)
                if not result.valid:
                    joined = "; ".join(result.errors)
                    raise ValueError(f"Invalid grasp trial in {path} record {index}: {joined}")
            records.append(record)
    return records


def summarize_grasp_trials(records: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    grasp_counts: Counter[str] = Counter()
    success_by_grasp: dict[str, list[float]] = defaultdict(list)

    for record in records:
        class_name = record["object"]["class_name"]
        grasp_type = record["candidate_grasp"]["grasp_type"]
        success_score = float(record["outcome"]["success_score"])

        class_counts[class_name] += 1
        grasp_counts[grasp_type] += 1
        success_by_grasp[grasp_type].append(success_score)

    mean_success_by_grasp = {
        grasp_type: sum(scores) / len(scores)
        for grasp_type, scores in success_by_grasp.items()
    }

    return {
        "num_trials": len(records),
        "class_counts": dict(class_counts),
        "grasp_counts": dict(grasp_counts),
        "mean_success_by_grasp": mean_success_by_grasp,
    }
