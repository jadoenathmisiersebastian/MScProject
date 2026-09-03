import numpy as np

from src.depth_candidate_verifier import (
    DepthCandidateFeatures,
    DepthVerifierConfig,
    extract_depth_candidate_features,
    verify_depth_candidate,
)


def test_unreliable_depth_falls_back_to_rgb():
    features = DepthCandidateFeatures(
        mask_pixels=100,
        valid_pixels=0,
        valid_fraction=0.0,
        median_depth_m=None,
        robust_depth_span_m=None,
        background_median_depth_m=None,
        foreground_background_contrast_m=None,
        physical_width_m=None,
        physical_height_m=None,
    )
    result = verify_depth_candidate(features)
    assert result.accepted
    assert result.status == "depth_unreliable_rgb_fallback"


def test_strong_physical_contradiction_is_rejected():
    features = DepthCandidateFeatures(
        mask_pixels=100,
        valid_pixels=100,
        valid_fraction=1.0,
        median_depth_m=1.0,
        robust_depth_span_m=0.1,
        background_median_depth_m=1.2,
        foreground_background_contrast_m=0.2,
        physical_width_m=0.9,
        physical_height_m=0.4,
    )
    result = verify_depth_candidate(features)
    assert not result.accepted
    assert "physical_extent_too_large" in result.reasons


def test_noisy_depth_span_alone_does_not_override_rgb():
    features = DepthCandidateFeatures(
        mask_pixels=100,
        valid_pixels=100,
        valid_fraction=1.0,
        median_depth_m=0.8,
        robust_depth_span_m=0.8,
        background_median_depth_m=1.0,
        foreground_background_contrast_m=0.2,
        physical_width_m=0.1,
        physical_height_m=0.2,
    )
    result = verify_depth_candidate(features)
    assert result.accepted


def test_feature_extraction_recovers_compact_object_geometry():
    depth = np.full((20, 20), 1.2, dtype=np.float32)
    depth[6:14, 7:13] = 0.8
    features = extract_depth_candidate_features(
        depth_m=depth,
        rgb_size=(200, 200),
        bbox_xyxy=(70, 60, 130, 140),
        polygon_xy=None,
        depth_intrinsics=np.asarray([[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]]),
    )
    assert features.valid_fraction == 1.0
    assert abs(features.median_depth_m - 0.8) < 1e-5
    assert features.foreground_background_contrast_m > 0.3
    assert verify_depth_candidate(features, DepthVerifierConfig()).accepted
