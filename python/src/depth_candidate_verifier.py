from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthVerifierConfig:
    minimum_valid_pixels: int = 24
    minimum_valid_fraction: float = 0.20
    maximum_physical_extent_m: float = 0.55
    maximum_robust_depth_span_m: float = 0.45
    maximum_behind_background_m: float = 0.08
    ring_radius_pixels: int = 4


@dataclass(frozen=True)
class DepthCandidateFeatures:
    mask_pixels: int
    valid_pixels: int
    valid_fraction: float
    median_depth_m: float | None
    robust_depth_span_m: float | None
    background_median_depth_m: float | None
    foreground_background_contrast_m: float | None
    physical_width_m: float | None
    physical_height_m: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True)
class DepthVerification:
    accepted: bool
    status: str
    reasons: tuple[str, ...]
    features: DepthCandidateFeatures

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "status": self.status,
            "reasons": list(self.reasons),
            "features": self.features.to_dict(),
        }


def _scaled_polygon_mask(
    depth_shape: tuple[int, int],
    rgb_size: tuple[int, int],
    bbox_xyxy: Sequence[float],
    polygon_xy: np.ndarray | None,
) -> np.ndarray:
    depth_height, depth_width = depth_shape
    rgb_width, rgb_height = rgb_size
    if rgb_width <= 0 or rgb_height <= 0:
        raise ValueError("RGB dimensions must be positive.")

    mask = np.zeros(depth_shape, dtype=np.uint8)
    if polygon_xy is not None and len(polygon_xy) >= 3:
        polygon = np.asarray(polygon_xy, dtype=np.float32).copy()
        polygon[:, 0] *= depth_width / rgb_width
        polygon[:, 1] *= depth_height / rgb_height
        polygon = np.rint(polygon).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)
    else:
        left, top, right, bottom = map(float, bbox_xyxy)
        left = int(np.clip(np.floor(left * depth_width / rgb_width), 0, depth_width - 1))
        top = int(np.clip(np.floor(top * depth_height / rgb_height), 0, depth_height - 1))
        right = int(np.clip(np.ceil(right * depth_width / rgb_width), left + 1, depth_width))
        bottom = int(np.clip(np.ceil(bottom * depth_height / rgb_height), top + 1, depth_height))
        mask[top:bottom, left:right] = 1
    return mask.astype(bool)


def extract_depth_candidate_features(
    depth_m: np.ndarray,
    rgb_size: tuple[int, int],
    bbox_xyxy: Sequence[float],
    depth_intrinsics: np.ndarray,
    polygon_xy: np.ndarray | None = None,
    ring_radius_pixels: int = 4,
) -> DepthCandidateFeatures:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected a two-dimensional depth image, got {depth.shape}.")
    intrinsics = np.asarray(depth_intrinsics, dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError("depth_intrinsics must be a 3x3 matrix.")

    mask = _scaled_polygon_mask(depth.shape, rgb_size, bbox_xyxy, polygon_xy)
    mask_pixels = int(np.count_nonzero(mask))
    valid_depth = np.isfinite(depth) & (depth > 0.0)

    # Erosion reduces foreground/background bleeding at the low-resolution LiDAR edge.
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    if np.count_nonzero(eroded) >= 8:
        sample_mask = eroded
    else:
        sample_mask = mask
    foreground_valid = sample_mask & valid_depth
    foreground_values = depth[foreground_valid]
    valid_pixels = int(foreground_values.size)
    valid_fraction = valid_pixels / max(int(np.count_nonzero(sample_mask)), 1)

    median_depth = None
    robust_span = None
    if valid_pixels:
        lower, median, upper = np.quantile(foreground_values, [0.05, 0.50, 0.95])
        median_depth = float(median)
        robust_span = float(upper - lower)

    radius = max(int(ring_radius_pixels), 1)
    kernel_size = radius * 2 + 1
    dilated = cv2.dilate(
        mask.astype(np.uint8),
        np.ones((kernel_size, kernel_size), np.uint8),
        iterations=1,
    ).astype(bool)
    ring = dilated & ~mask & valid_depth
    ring_values = depth[ring]
    background_median = float(np.median(ring_values)) if ring_values.size >= 8 else None
    contrast = (
        background_median - median_depth
        if background_median is not None and median_depth is not None
        else None
    )

    physical_width = None
    physical_height = None
    if median_depth is not None:
        rgb_width, rgb_height = rgb_size
        depth_height, depth_width = depth.shape
        bbox_width_depth = max(float(bbox_xyxy[2]) - float(bbox_xyxy[0]), 0.0) * depth_width / rgb_width
        bbox_height_depth = max(float(bbox_xyxy[3]) - float(bbox_xyxy[1]), 0.0) * depth_height / rgb_height
        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        if fx > 0.0 and fy > 0.0:
            physical_width = float(bbox_width_depth * median_depth / fx)
            physical_height = float(bbox_height_depth * median_depth / fy)

    return DepthCandidateFeatures(
        mask_pixels=mask_pixels,
        valid_pixels=valid_pixels,
        valid_fraction=float(valid_fraction),
        median_depth_m=median_depth,
        robust_depth_span_m=robust_span,
        background_median_depth_m=background_median,
        foreground_background_contrast_m=contrast,
        physical_width_m=physical_width,
        physical_height_m=physical_height,
    )


def verify_depth_candidate(
    features: DepthCandidateFeatures,
    config: DepthVerifierConfig | None = None,
) -> DepthVerification:
    config = config or DepthVerifierConfig()
    if (
        features.valid_pixels < config.minimum_valid_pixels
        or features.valid_fraction < config.minimum_valid_fraction
        or features.median_depth_m is None
    ):
        return DepthVerification(
            accepted=True,
            status="depth_unreliable_rgb_fallback",
            reasons=("insufficient_reliable_depth",),
            features=features,
        )

    reasons: list[str] = []
    extents = [
        value
        for value in (features.physical_width_m, features.physical_height_m)
        if value is not None
    ]
    extent_too_large = bool(
        extents and max(extents) > config.maximum_physical_extent_m
    )
    behind_background = bool(
        features.foreground_background_contrast_m is not None
        and features.foreground_background_contrast_m
        < -config.maximum_behind_background_m
    )
    if extent_too_large:
        reasons.append("physical_extent_too_large")
    if (
        (extent_too_large or behind_background)
        and
        features.robust_depth_span_m is not None
        and features.robust_depth_span_m > config.maximum_robust_depth_span_m
    ):
        reasons.append("depth_span_too_large")
    if behind_background:
        reasons.append("candidate_behind_local_background")

    if reasons:
        return DepthVerification(
            accepted=False,
            status="rejected_strong_depth_contradiction",
            reasons=tuple(reasons),
            features=features,
        )
    return DepthVerification(
        accepted=True,
        status="accepted_depth_plausible",
        reasons=(),
        features=features,
    )
