using System.Collections.Generic;
using UnityEngine;

public class TabletopCameraRandomizer : MonoBehaviour
{
    private struct VisibilityBlockerBounds
    {
        public Bounds bounds;
        public string name;
    }

    [Header("References")]
    public Transform cameraTransform;
    public Transform cameraTarget;
    public Transform validCameraRegion;

    [Header("Target Jitter")]
    public Vector2 targetXJitter = new Vector2(-0.1f, 0.1f);
    public Vector2 targetYJitter = new Vector2(0.0f, 0.15f);
    public Vector2 targetZJitter = new Vector2(-0.1f, 0.1f);

    [Header("Camera Distance")]
    public float minTargetDistance = 0.45f;
    public float maxTargetDistance = 2.25f;
    public int maxCameraPlacementAttempts = 50;

    [Header("Camera Collision")]
    public LayerMask cameraCollisionMask = ~0;
    [Min(0f)] public float cameraClearanceRadius = 0.12f;
    [Min(0f)] public float minimumForwardClearance = 0.55f;
    [Min(0f)] public float forwardClearanceRadius = 0.06f;

    [Header("Null Sample Views")]
    [Tooltip("Optional volume used for null-sample look targets. Falls back to Valid Camera Region when unassigned.")]
    public Transform nullLookRegion;
    [Min(0f)] public float minNullLookDistance = 0.5f;
    [Min(0f)] public float maxNullLookDistance = 3.0f;

    [Header("Line Of Sight")]
    public bool requireLineOfSight = false;
    public LayerMask lineOfSightMask = ~0;
    public float lineOfSightTargetRadius = 0.08f;

    [Header("Full Target Visibility")]
    [Range(0f, 0.2f)] public float targetViewportMargin = 0.02f;
    [Range(0f, 0.45f)] public float visibilityBoundsInset = 0.12f;
    [Min(0f)] public float visibilityRayTolerance = 0.015f;
    [Min(0f)] public float nearClipMargin = 0.01f;

    public void RandomizeCamera()
    {
        if (cameraTarget == null)
        {
            Debug.LogError("Camera target is missing.");
            return;
        }

        RandomizeCamera(cameraTarget.position);
    }

    public void RandomizeCamera(Vector3 targetPosition)
    {
        if (!TryRandomizeCamera(targetPosition, out string rejectionReason))
        {
            Debug.LogWarning($"Could not find valid camera pose. Reason: {rejectionReason}");
        }
    }

    public bool TryRandomizeCamera(Vector3 targetPosition, out string rejectionReason)
    {
        rejectionReason = null;

        if (!TryGetCameraRegion(out Bounds region, out rejectionReason))
        {
            return false;
        }

        for (int attempt = 0; attempt < maxCameraPlacementAttempts; attempt++)
        {
            Vector3 cameraPosition = SamplePoint(region);

            if (!IsCameraPositionClear(cameraPosition, out rejectionReason))
            {
                continue;
            }

            float distance = Vector3.Distance(cameraPosition, targetPosition);

            if (!IsDistanceAllowed(distance, minTargetDistance, maxTargetDistance))
            {
                rejectionReason =
                    $"Distance {distance:F2}m outside allowed range " +
                    $"[{minTargetDistance:F2}, {maxTargetDistance:F2}]m.";
                continue;
            }

            Vector3 jitteredTargetPosition = GetJitteredTarget(targetPosition);

            if (requireLineOfSight && IsLineOfSightBlocked(cameraPosition, jitteredTargetPosition))
            {
                rejectionReason = "Line of sight to target is blocked.";
                continue;
            }

            ApplyCameraPose(cameraPosition, jitteredTargetPosition);
            return true;
        }

        if (rejectionReason == null)
        {
            rejectionReason = "No camera pose sampled.";
        }

        return false;
    }

    public bool TryRandomizeCamera(
        GameObject targetObject,
        Camera captureCamera,
        out string rejectionReason
    )
    {
        rejectionReason = null;

        if (targetObject == null)
        {
            rejectionReason = "Target object is missing.";
            return false;
        }

        if (captureCamera == null)
        {
            rejectionReason = "Capture camera is missing.";
            return false;
        }

        if (!TryGetObjectBounds(targetObject, out Bounds targetBounds))
        {
            rejectionReason = $"Target object {targetObject.name} has no renderers.";
            return false;
        }

        if (!TryGetCameraRegion(out Bounds region, out rejectionReason))
        {
            return false;
        }

        for (int attempt = 0; attempt < maxCameraPlacementAttempts; attempt++)
        {
            Vector3 cameraPosition = SamplePoint(region);

            if (!IsCameraPositionClear(cameraPosition, out rejectionReason))
            {
                continue;
            }

            float distance = Vector3.Distance(cameraPosition, targetBounds.center);

            if (!IsDistanceAllowed(distance, minTargetDistance, maxTargetDistance))
            {
                rejectionReason =
                    $"Distance {distance:F2}m outside allowed range " +
                    $"[{minTargetDistance:F2}, {maxTargetDistance:F2}]m.";
                continue;
            }

            ApplyCameraPose(cameraPosition, GetJitteredTarget(targetBounds.center));
            Physics.SyncTransforms();

            if (!IsTargetFullyVisible(targetObject, captureCamera, out rejectionReason))
            {
                continue;
            }

            return true;
        }

        if (rejectionReason == null)
        {
            rejectionReason = "No fully visible target camera pose could be sampled.";
        }

        return false;
    }

    public bool TryRandomizeNullCamera(out string rejectionReason)
    {
        rejectionReason = null;

        if (!TryGetCameraRegion(out Bounds cameraRegion, out rejectionReason))
        {
            return false;
        }

        Transform lookRegionTransform = nullLookRegion != null
            ? nullLookRegion
            : validCameraRegion;

        Bounds lookRegion = GetRegionBounds(lookRegionTransform);

        for (int attempt = 0; attempt < maxCameraPlacementAttempts; attempt++)
        {
            Vector3 cameraPosition = SamplePoint(cameraRegion);

            if (!IsCameraPositionClear(cameraPosition, out rejectionReason))
            {
                continue;
            }

            Vector3 lookPosition = SamplePoint(lookRegion);
            float lookDistance = Vector3.Distance(cameraPosition, lookPosition);

            if (!IsDistanceAllowed(lookDistance, minNullLookDistance, maxNullLookDistance))
            {
                rejectionReason =
                    $"Null look distance {lookDistance:F2}m outside allowed range " +
                    $"[{minNullLookDistance:F2}, {maxNullLookDistance:F2}]m.";
                continue;
            }

            ApplyCameraPose(cameraPosition, lookPosition);
            Physics.SyncTransforms();

            if (!IsCurrentCameraPoseClear(out rejectionReason))
            {
                continue;
            }

            return true;
        }

        if (rejectionReason == null)
        {
            rejectionReason = "No collision-free null-sample room view could be sampled.";
        }

        return false;
    }

    public bool IsCurrentCameraDistanceValid(Vector3 targetPosition, out float distance)
    {
        distance = 0f;

        if (cameraTransform == null)
        {
            return false;
        }

        distance = Vector3.Distance(cameraTransform.position, targetPosition);
        return distance >= minTargetDistance && distance <= maxTargetDistance;
    }

    public bool IsCurrentCameraPoseClear(out string rejectionReason)
    {
        rejectionReason = null;

        if (cameraTransform == null)
        {
            rejectionReason = "Camera transform is missing.";
            return false;
        }

        if (!IsCameraPositionClear(cameraTransform.position, out rejectionReason))
        {
            return false;
        }

        if (minimumForwardClearance <= 0f)
        {
            return true;
        }

        Ray ray = new Ray(cameraTransform.position, cameraTransform.forward);

        if (Physics.SphereCast(
            ray,
            forwardClearanceRadius,
            out RaycastHit hit,
            minimumForwardClearance,
            cameraCollisionMask,
            QueryTriggerInteraction.Ignore
        ))
        {
            rejectionReason =
                $"Camera view is immediately blocked by {hit.collider.name} " +
                $"at {hit.distance:F2}m.";
            return false;
        }

        return true;
    }

    public bool IsTargetFullyVisible(
        GameObject targetObject,
        Camera captureCamera,
        out string rejectionReason
    )
    {
        rejectionReason = null;

        if (targetObject == null || captureCamera == null)
        {
            rejectionReason = "Target object or capture camera is missing.";
            return false;
        }

        if (!TryGetObjectBounds(targetObject, out Bounds targetBounds))
        {
            rejectionReason = $"Target object {targetObject.name} has no renderers.";
            return false;
        }

        Vector3[] visibilityPoints = BuildVisibilityPoints(targetBounds);
        VisibilityBlockerBounds[] uncollidedWallFeatures =
            GetUncollidedWallFeatureBounds();

        foreach (Vector3 point in visibilityPoints)
        {
            Vector3 viewportPoint = captureCamera.WorldToViewportPoint(point);
            float minimumDepth = captureCamera.nearClipPlane + nearClipMargin;

            if (viewportPoint.z < minimumDepth ||
                viewportPoint.x < targetViewportMargin ||
                viewportPoint.x > 1f - targetViewportMargin ||
                viewportPoint.y < targetViewportMargin ||
                viewportPoint.y > 1f - targetViewportMargin)
            {
                rejectionReason = "Target bounds are clipped or outside the camera view.";
                return false;
            }

            if (!IsVisibilityRayClear(
                captureCamera.transform.position,
                point,
                targetObject,
                uncollidedWallFeatures,
                out string blockerName
            ))
            {
                rejectionReason = $"Target is occluded by {blockerName}.";
                return false;
            }
        }

        return true;
    }

    private bool TryGetCameraRegion(out Bounds region, out string rejectionReason)
    {
        region = default;
        rejectionReason = null;

        if (cameraTransform == null || validCameraRegion == null)
        {
            rejectionReason = "Camera transform or valid camera region is missing.";
            Debug.LogError("Camera randomizer references are missing.");
            return false;
        }

        region = GetRegionBounds(validCameraRegion);
        return true;
    }

    private Bounds GetRegionBounds(Transform regionTransform)
    {
        return new Bounds(regionTransform.position, regionTransform.lossyScale);
    }

    private Vector3 SamplePoint(Bounds bounds)
    {
        return new Vector3(
            Random.Range(bounds.min.x, bounds.max.x),
            Random.Range(bounds.min.y, bounds.max.y),
            Random.Range(bounds.min.z, bounds.max.z)
        );
    }

    private bool IsCameraPositionClear(Vector3 cameraPosition, out string rejectionReason)
    {
        rejectionReason = null;

        if (cameraClearanceRadius <= 0f)
        {
            return true;
        }

        Collider[] overlaps = Physics.OverlapSphere(
            cameraPosition,
            cameraClearanceRadius,
            cameraCollisionMask,
            QueryTriggerInteraction.Ignore
        );

        foreach (Collider overlap in overlaps)
        {
            if (overlap == null || IsPartOfCamera(overlap.transform))
            {
                continue;
            }

            rejectionReason = $"Camera position overlaps {overlap.name}.";
            return false;
        }

        return true;
    }

    private bool IsVisibilityRayClear(
        Vector3 origin,
        Vector3 targetPoint,
        GameObject targetObject,
        VisibilityBlockerBounds[] uncollidedWallFeatures,
        out string blockerName
    )
    {
        blockerName = null;
        Vector3 direction = targetPoint - origin;
        float pointDistance = direction.magnitude;

        if (pointDistance <= visibilityRayTolerance)
        {
            blockerName = "camera proximity";
            return false;
        }

        RaycastHit[] hits = Physics.RaycastAll(
            origin,
            direction.normalized,
            pointDistance + visibilityRayTolerance,
            lineOfSightMask,
            QueryTriggerInteraction.Ignore
        );

        float nearestTargetDistance = float.PositiveInfinity;
        float nearestBlockerDistance = float.PositiveInfinity;
        string nearestBlockerName = null;

        foreach (RaycastHit hit in hits)
        {
            if (hit.collider == null || IsPartOfCamera(hit.collider.transform))
            {
                continue;
            }

            if (hit.collider.transform == targetObject.transform ||
                hit.collider.transform.IsChildOf(targetObject.transform))
            {
                nearestTargetDistance = Mathf.Min(nearestTargetDistance, hit.distance);
                continue;
            }

            if (hit.distance < nearestBlockerDistance)
            {
                nearestBlockerDistance = hit.distance;
                nearestBlockerName = hit.collider.name;
            }
        }

        float clearDistance = float.IsPositiveInfinity(nearestTargetDistance)
            ? pointDistance
            : nearestTargetDistance;

        if (nearestBlockerDistance < clearDistance - visibilityRayTolerance)
        {
            blockerName = nearestBlockerName ?? "scene geometry";
            return false;
        }

        if (IsBlockedByUncollidedWallFeature(
            new Ray(origin, direction.normalized),
            clearDistance,
            uncollidedWallFeatures,
            out blockerName
        ))
        {
            return false;
        }

        return true;
    }

    private bool IsBlockedByUncollidedWallFeature(
        Ray ray,
        float clearDistance,
        VisibilityBlockerBounds[] blockerBounds,
        out string blockerName
    )
    {
        blockerName = null;

        foreach (VisibilityBlockerBounds blocker in blockerBounds)
        {
            if (blocker.bounds.IntersectRay(ray, out float hitDistance) &&
                hitDistance < clearDistance - visibilityRayTolerance)
            {
                blockerName = blocker.name;
                return true;
            }
        }

        return false;
    }

    private VisibilityBlockerBounds[] GetUncollidedWallFeatureBounds()
    {
        List<VisibilityBlockerBounds> blockerBounds =
            new List<VisibilityBlockerBounds>();

        WallMountedFootprint[] footprints =
            Object.FindObjectsByType<WallMountedFootprint>(FindObjectsInactive.Exclude);

        foreach (WallMountedFootprint footprint in footprints)
        {
            Transform featureRoot = FindWallFeatureRoot(footprint.transform);

            if (featureRoot == null || HasActiveBlockingCollider(featureRoot))
            {
                continue;
            }

            Renderer[] renderers = featureRoot.GetComponentsInChildren<Renderer>();

            if (!TryGetRendererBounds(renderers, out Bounds bounds))
            {
                continue;
            }

            blockerBounds.Add(new VisibilityBlockerBounds
            {
                bounds = bounds,
                name = featureRoot.name
            });
        }

        return blockerBounds.ToArray();
    }

    private bool HasActiveBlockingCollider(Transform root)
    {
        Collider[] colliders = root.GetComponentsInChildren<Collider>();

        foreach (Collider collider in colliders)
        {
            if (collider.enabled && !collider.isTrigger)
            {
                return true;
            }
        }

        return false;
    }

    private Transform FindWallFeatureRoot(Transform child)
    {
        Transform current = child;

        while (current != null)
        {
            if (current.name.StartsWith("WallFeature_"))
            {
                return current;
            }

            current = current.parent;
        }

        return null;
    }

    private Vector3[] BuildVisibilityPoints(Bounds bounds)
    {
        float insetScale = 1f - Mathf.Clamp(visibilityBoundsInset, 0f, 0.45f);
        Vector3 extent = bounds.extents * insetScale;
        Vector3 center = bounds.center;

        return new[]
        {
            center,
            center + new Vector3(-extent.x, -extent.y, -extent.z),
            center + new Vector3(-extent.x, -extent.y, extent.z),
            center + new Vector3(-extent.x, extent.y, -extent.z),
            center + new Vector3(-extent.x, extent.y, extent.z),
            center + new Vector3(extent.x, -extent.y, -extent.z),
            center + new Vector3(extent.x, -extent.y, extent.z),
            center + new Vector3(extent.x, extent.y, -extent.z),
            center + new Vector3(extent.x, extent.y, extent.z),
            center + new Vector3(-extent.x, 0f, 0f),
            center + new Vector3(extent.x, 0f, 0f),
            center + new Vector3(0f, -extent.y, 0f),
            center + new Vector3(0f, extent.y, 0f),
            center + new Vector3(0f, 0f, -extent.z),
            center + new Vector3(0f, 0f, extent.z)
        };
    }

    private bool TryGetObjectBounds(GameObject obj, out Bounds bounds)
    {
        Renderer[] renderers = obj.GetComponentsInChildren<Renderer>();

        return TryGetRendererBounds(renderers, out bounds);
    }

    private bool TryGetRendererBounds(Renderer[] renderers, out Bounds bounds)
    {
        bounds = default;

        if (renderers.Length == 0)
        {
            return false;
        }

        bounds = renderers[0].bounds;

        for (int i = 1; i < renderers.Length; i++)
        {
            bounds.Encapsulate(renderers[i].bounds);
        }

        return true;
    }

    private bool IsPartOfCamera(Transform candidate)
    {
        if (candidate == null || cameraTransform == null)
        {
            return false;
        }

        return candidate == cameraTransform ||
               candidate.IsChildOf(cameraTransform) ||
               cameraTransform.IsChildOf(candidate);
    }

    private bool IsDistanceAllowed(float distance, float minimum, float maximum)
    {
        return distance >= minimum && distance <= maximum;
    }

    private Vector3 GetJitteredTarget(Vector3 targetPosition)
    {
        return targetPosition + new Vector3(
            Random.Range(targetXJitter.x, targetXJitter.y),
            Random.Range(targetYJitter.x, targetYJitter.y),
            Random.Range(targetZJitter.x, targetZJitter.y)
        );
    }

    private void ApplyCameraPose(Vector3 cameraPosition, Vector3 targetPosition)
    {
        cameraTransform.position = cameraPosition;
        cameraTransform.LookAt(targetPosition);
    }

    private bool IsLineOfSightBlocked(Vector3 cameraPosition, Vector3 targetPosition)
    {
        Vector3 direction = targetPosition - cameraPosition;
        float distance = direction.magnitude;

        if (distance <= lineOfSightTargetRadius)
        {
            return false;
        }

        Ray ray = new Ray(cameraPosition, direction.normalized);
        float rayDistance = Mathf.Max(0f, distance - lineOfSightTargetRadius);

        return Physics.Raycast(ray, rayDistance, lineOfSightMask, QueryTriggerInteraction.Ignore);
    }
}
