from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from .utils import natural_step_number


@dataclass(frozen=True)
class UnityObjectAnnotation:
    label: str
    origin: tuple[float, float]
    dimension: tuple[float, float]
    instance_id: int | None = None
    label_id: int | None = None


@dataclass(frozen=True)
class UnityFrame:
    frame_json: Path
    image_path: Path
    width: float
    height: float
    objects: list[UnityObjectAnnotation]
    dataset_name: str
    sequence_name: str
    step_name: str


def dataset_path_from_name(unity_output_root: Path, dataset: str | Path) -> Path:
    dataset = Path(dataset)
    if dataset.is_absolute():
        return dataset.expanduser().resolve()
    return (unity_output_root / dataset).resolve()


def find_frame_jsons(dataset_path: Path) -> list[Path]:
    frame_jsons = sorted(
        dataset_path.glob("sequence.*/step*.frame_data.json"),
        key=lambda p: (p.parent.name, natural_step_number(p)),
    )
    if not frame_jsons:
        raise FileNotFoundError(f"No step*.frame_data.json files found below: {dataset_path}")
    return frame_jsons


def _find_bbox_annotation(capture: dict) -> dict | None:
    for ann in capture.get("annotations", []):
        if "BoundingBox2DAnnotation" in ann.get("@type", ""):
            return ann
    return None


def read_unity_frame(frame_json: Path) -> UnityFrame | None:
    with frame_json.open("r") as f:
        data = json.load(f)

    captures = data.get("captures", [])
    if not captures:
        return None

    capture = captures[0]
    image_path = frame_json.parent / capture["filename"]
    if not image_path.exists():
        raise FileNotFoundError(f"Image referenced by {frame_json} does not exist: {image_path}")

    width, height = capture["dimension"]
    bbox_ann = _find_bbox_annotation(capture)
    if bbox_ann is None:
        objects = []
    else:
        objects = [
            UnityObjectAnnotation(
                label=str(obj["labelName"]),
                origin=(float(obj["origin"][0]), float(obj["origin"][1])),
                dimension=(float(obj["dimension"][0]), float(obj["dimension"][1])),
                instance_id=obj.get("instanceId"),
                label_id=obj.get("labelId"),
            )
            for obj in bbox_ann.get("values", [])
        ]

    dataset_path = frame_json.parents[1]
    return UnityFrame(
        frame_json=frame_json,
        image_path=image_path,
        width=float(width),
        height=float(height),
        objects=objects,
        dataset_name=dataset_path.name,
        sequence_name=frame_json.parent.name,
        step_name=frame_json.name.split(".", 1)[0],
    )


def iter_unity_frames(dataset_path: Path) -> list[UnityFrame]:
    frames: list[UnityFrame] = []
    for frame_json in find_frame_jsons(dataset_path):
        frame = read_unity_frame(frame_json)
        if frame is not None:
            frames.append(frame)
    return frames
