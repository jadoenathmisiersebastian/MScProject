from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product


GRASP_TYPES = ("power", "cylindrical", "pinch")
WRIST_ROLL_DEGREES = (-60, -30, 0, 30, 60)
HAND_APERTURES = {
    "small": 0.35,
    "medium": 0.60,
    "large": 0.85,
}
DEFAULT_APPROACH_DIRECTION_CAMERA = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class GraspCandidate:
    candidate_id: str
    grasp_type: str
    wrist_roll_degrees: int
    hand_aperture: float
    hand_aperture_label: str
    approach_direction_camera: tuple[float, float, float]

    def to_dict(self) -> dict:
        return asdict(self)


def generate_grasp_candidates(
    grasp_types: tuple[str, ...] = GRASP_TYPES,
    wrist_roll_degrees: tuple[int, ...] = WRIST_ROLL_DEGREES,
    hand_apertures: dict[str, float] = HAND_APERTURES,
    approach_direction_camera: tuple[float, float, float] = DEFAULT_APPROACH_DIRECTION_CAMERA,
) -> list[GraspCandidate]:
    candidates: list[GraspCandidate] = []

    for grasp_type, wrist_roll, aperture_label in product(grasp_types, wrist_roll_degrees, hand_apertures):
        aperture = hand_apertures[aperture_label]
        candidate_id = f"{grasp_type}_roll{wrist_roll:+d}_{aperture_label}"
        candidates.append(
            GraspCandidate(
                candidate_id=candidate_id,
                grasp_type=grasp_type,
                wrist_roll_degrees=wrist_roll,
                hand_aperture=aperture,
                hand_aperture_label=aperture_label,
                approach_direction_camera=approach_direction_camera,
            )
        )

    return candidates


def candidates_as_dicts() -> list[dict]:
    return [candidate.to_dict() for candidate in generate_grasp_candidates()]
