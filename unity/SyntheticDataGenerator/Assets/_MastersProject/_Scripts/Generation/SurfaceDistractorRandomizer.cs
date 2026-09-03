using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class WeightedSurfaceDistractorPrefab
{
    public GameObject prefab;
    public float weight = 1f;

    [Tooltip("0 disables this prefab, -1 means unlimited, 1 means at most one per generated scene.")]
    public int maxInstancesPerScene = 99;

    [Tooltip("Optional support surface IDs this distractor may spawn on. Leave empty to allow any non-selected support surface.")]
    public string[] allowedSurfaceIds;

    public bool CanSpawnOn(SupportSurface surface)
    {
        if (surface == null)
        {
            return false;
        }

        if (allowedSurfaceIds == null || allowedSurfaceIds.Length == 0)
        {
            return true;
        }

        foreach (string allowedSurfaceId in allowedSurfaceIds)
        {
            if (string.IsNullOrWhiteSpace(allowedSurfaceId))
            {
                continue;
            }

            if (surface.surfaceId == allowedSurfaceId)
            {
                return true;
            }
        }

        return false;
    }
}

public class SurfaceDistractorRandomizer : MonoBehaviour
{
    [Header("Spawn Parent")]
    public Transform surfaceDistractorsParent;

    [Header("Distractor Prefabs")]
    public WeightedSurfaceDistractorPrefab[] distractorPrefabs;

    [Header("Counts")]
    public int minDistractorCount = 0;
    public int maxDistractorCount = 4;
    public int maxAttemptsPerDistractor = 50;

    [Header("Placement")]
    public bool useSupportSurfaceWeights = true;
    public bool randomizeYaw = true;
    public Vector2 yawRange = new Vector2(0f, 360f);

    [Header("Debug")]
    public bool logPlacementSummary = false;
    public bool logPlacementWarnings = true;

    private class PlacedDistractor
    {
        public SupportSurface surface;
        public SurfaceDistractorFootprint footprint;
    }

    private readonly List<GameObject> spawnedDistractors = new List<GameObject>();
    private readonly List<PlacedDistractor> placedDistractors = new List<PlacedDistractor>();
    private readonly Dictionary<GameObject, int> placedPrefabCounts = new Dictionary<GameObject, int>();

    public void RandomizeSurfaceDistractors(
        SupportSurfaceRegistry supportSurfaceRegistry,
        SupportSurface excludedSurface
    )
    {
        ClearDistractors();

        if (supportSurfaceRegistry == null)
        {
            Debug.LogError("Support surface registry is not assigned.");
            return;
        }

        if (surfaceDistractorsParent == null)
        {
            Debug.LogError("Surface distractors parent is not assigned.");
            return;
        }

        if (distractorPrefabs == null || distractorPrefabs.Length == 0)
        {
            if (logPlacementWarnings)
            {
                Debug.LogWarning("No surface distractor prefabs assigned.");
            }

            return;
        }

        supportSurfaceRegistry.Refresh();

        int targetCount = Random.Range(
            Mathf.Max(0, minDistractorCount),
            Mathf.Max(minDistractorCount, maxDistractorCount) + 1
        );

        int spawnedCount = 0;

        for (int i = 0; i < targetCount; i++)
        {
            if (TrySpawnOneDistractor(supportSurfaceRegistry, excludedSurface))
            {
                spawnedCount++;
            }
        }

        if (logPlacementSummary)
        {
            string excludedId = excludedSurface != null ? excludedSurface.surfaceId : "<none>";
            Debug.Log(
                $"Spawned {spawnedCount}/{targetCount} surface distractors. " +
                $"Excluded target surface: {excludedId}."
            );
        }
    }

    public void ClearDistractors()
    {
        for (int i = spawnedDistractors.Count - 1; i >= 0; i--)
        {
            GameObject distractor = spawnedDistractors[i];

            if (distractor == null)
            {
                continue;
            }

            distractor.SetActive(false);
            Destroy(distractor);
        }

        spawnedDistractors.Clear();
        placedDistractors.Clear();
        placedPrefabCounts.Clear();
    }

    private bool TrySpawnOneDistractor(
        SupportSurfaceRegistry supportSurfaceRegistry,
        SupportSurface excludedSurface
    )
    {
        for (int attempt = 0; attempt < maxAttemptsPerDistractor; attempt++)
        {
            WeightedSurfaceDistractorPrefab selectedPrefab = GetRandomAvailablePrefab();

            if (selectedPrefab == null || selectedPrefab.prefab == null)
            {
                return false;
            }

            SupportSurface surface = GetRandomAllowedSurface(
                supportSurfaceRegistry.AvailableSurfaces,
                excludedSurface,
                selectedPrefab
            );

            if (surface == null)
            {
                return false;
            }

            GameObject candidate = Instantiate(
                selectedPrefab.prefab,
                Vector3.zero,
                Quaternion.identity,
                surfaceDistractorsParent
            );

            candidate.SetActive(false);

            SurfaceDistractorFootprint footprint =
                candidate.GetComponentInChildren<SurfaceDistractorFootprint>(true);

            if (footprint == null)
            {
                Debug.LogError(
                    $"Surface distractor prefab {selectedPrefab.prefab.name} " +
                    "has no SurfaceDistractorFootprint component."
                );

                Destroy(candidate);
                return false;
            }

            PlaceCandidateOnSurface(candidate, footprint, surface);

            if (!IsFootprintValidOnSurface(footprint, surface))
            {
                Destroy(candidate);
                continue;
            }

            candidate.name = $"SurfaceDistractor_{surface.surfaceId}_{selectedPrefab.prefab.name}";
            candidate.SetActive(true);

            spawnedDistractors.Add(candidate);
            placedDistractors.Add(new PlacedDistractor
            {
                surface = surface,
                footprint = footprint
            });

            IncrementPrefabCount(selectedPrefab.prefab);
            return true;
        }

        if (logPlacementWarnings)
        {
            Debug.LogWarning("Could not place requested surface distractor after max attempts.");
        }

        return false;
    }

    private void PlaceCandidateOnSurface(
        GameObject candidate,
        SurfaceDistractorFootprint footprint,
        SupportSurface surface
    )
    {
        float yaw = randomizeYaw ? Random.Range(yawRange.x, yawRange.y) : 0f;
        candidate.transform.rotation = surface.transform.rotation * Quaternion.Euler(0f, yaw, 0f);

        Vector2 paddedSize = footprint.PaddedSize;
        float halfAreaX = Mathf.Max(0f, surface.spawnAreaSize.x * 0.5f - paddedSize.x * 0.5f);
        float halfAreaZ = Mathf.Max(0f, surface.spawnAreaSize.y * 0.5f - paddedSize.y * 0.5f);

        float localX = Random.Range(-halfAreaX, halfAreaX);
        float localZ = Random.Range(-halfAreaZ, halfAreaZ);

        Vector3 desiredFootprintPosition = surface.transform.TransformPoint(
            new Vector3(localX, footprint.surfaceHeightOffset, localZ)
        );

        Vector3 offset = desiredFootprintPosition - footprint.transform.position;
        candidate.transform.position += offset;
    }

    private bool IsFootprintValidOnSurface(
        SurfaceDistractorFootprint footprint,
        SupportSurface surface
    )
    {
        if (!IsFootprintInsideSurface(footprint, surface))
        {
            return false;
        }

        foreach (PlacedDistractor placedDistractor in placedDistractors)
        {
            if (placedDistractor.surface != surface)
            {
                continue;
            }

            if (DoFootprintsOverlapOnSurface(
                footprint,
                placedDistractor.footprint,
                surface
            ))
            {
                return false;
            }
        }

        return true;
    }

    private bool IsFootprintInsideSurface(
        SurfaceDistractorFootprint footprint,
        SupportSurface surface)
    {
        Vector3[] corners = footprint.GetWorldCorners(true);
        float allowedX = surface.spawnAreaSize.x * 0.5f - surface.edgeMargin;
        float allowedZ = surface.spawnAreaSize.y * 0.5f - surface.edgeMargin;

        if (allowedX <= 0f || allowedZ <= 0f)
        {
            return false;
        }

        foreach (Vector3 corner in corners)
        {
            Vector3 localCorner = surface.transform.InverseTransformPoint(corner);

            if (Mathf.Abs(localCorner.x) > allowedX ||
                Mathf.Abs(localCorner.z) > allowedZ)
            {
                return false;
            }
        }

        return true;
    }

    private bool DoFootprintsOverlapOnSurface(
        SurfaceDistractorFootprint a,
        SurfaceDistractorFootprint b,
        SupportSurface surface
    )
    {
        Vector2[] aPoints = GetSurfaceLocalPoints(a, surface);
        Vector2[] bPoints = GetSurfaceLocalPoints(b, surface);

        return !HasSeparatingAxis(aPoints, bPoints) &&
               !HasSeparatingAxis(bPoints, aPoints);
    }

    private Vector2[] GetSurfaceLocalPoints(
        SurfaceDistractorFootprint footprint,
        SupportSurface surface
    )
    {
        Vector3[] worldCorners = footprint.GetWorldCorners(true);
        Vector2[] points = new Vector2[worldCorners.Length];

        for (int i = 0; i < worldCorners.Length; i++)
        {
            Vector3 localCorner = surface.transform.InverseTransformPoint(worldCorners[i]);
            points[i] = new Vector2(localCorner.x, localCorner.z);
        }

        return points;
    }

    private bool HasSeparatingAxis(Vector2[] aPoints, Vector2[] bPoints)
    {
        for (int i = 0; i < aPoints.Length; i++)
        {
            Vector2 current = aPoints[i];
            Vector2 next = aPoints[(i + 1) % aPoints.Length];
            Vector2 edge = next - current;
            Vector2 axis = new Vector2(-edge.y, edge.x).normalized;

            ProjectOntoAxis(aPoints, axis, out float minA, out float maxA);
            ProjectOntoAxis(bPoints, axis, out float minB, out float maxB);

            if (maxA < minB || maxB < minA)
            {
                return true;
            }
        }

        return false;
    }

    private void ProjectOntoAxis(
        Vector2[] points,
        Vector2 axis,
        out float min,
        out float max
    )
    {
        min = Vector2.Dot(points[0], axis);
        max = min;

        for (int i = 1; i < points.Length; i++)
        {
            float projection = Vector2.Dot(points[i], axis);
            min = Mathf.Min(min, projection);
            max = Mathf.Max(max, projection);
        }
    }

    private WeightedSurfaceDistractorPrefab GetRandomAvailablePrefab()
    {
        float totalWeight = 0f;

        foreach (WeightedSurfaceDistractorPrefab distractorPrefab in distractorPrefabs)
        {
            if (!IsPrefabAvailable(distractorPrefab))
            {
                continue;
            }

            totalWeight += Mathf.Max(0f, distractorPrefab.weight);
        }

        if (totalWeight <= 0f)
        {
            return null;
        }

        float randomValue = Random.Range(0f, totalWeight);
        float cumulative = 0f;

        foreach (WeightedSurfaceDistractorPrefab distractorPrefab in distractorPrefabs)
        {
            if (!IsPrefabAvailable(distractorPrefab))
            {
                continue;
            }

            cumulative += Mathf.Max(0f, distractorPrefab.weight);

            if (randomValue <= cumulative)
            {
                return distractorPrefab;
            }
        }

        return null;
    }

    private bool IsPrefabAvailable(WeightedSurfaceDistractorPrefab distractorPrefab)
    {
        if (distractorPrefab == null || distractorPrefab.prefab == null)
        {
            return false;
        }

        if (distractorPrefab.weight <= 0f)
        {
            return false;
        }

        if (distractorPrefab.maxInstancesPerScene == 0)
        {
            return false;
        }

        if (distractorPrefab.maxInstancesPerScene < 0)
        {
            return true;
        }

        placedPrefabCounts.TryGetValue(distractorPrefab.prefab, out int count);
        return count < distractorPrefab.maxInstancesPerScene;
    }

    private SupportSurface GetRandomAllowedSurface(
        IReadOnlyList<SupportSurface> surfaces,
        SupportSurface excludedSurface,
        WeightedSurfaceDistractorPrefab distractorPrefab
    )
    {
        float totalWeight = 0f;

        foreach (SupportSurface surface in surfaces)
        {
            if (!IsSurfaceAllowed(surface, excludedSurface, distractorPrefab))
            {
                continue;
            }

            totalWeight += useSupportSurfaceWeights ? Mathf.Max(0f, surface.selectionWeight) : 1f;
        }

        if (totalWeight <= 0f)
        {
            return null;
        }

        float randomValue = Random.Range(0f, totalWeight);
        float cumulative = 0f;

        foreach (SupportSurface surface in surfaces)
        {
            if (!IsSurfaceAllowed(surface, excludedSurface, distractorPrefab))
            {
                continue;
            }

            cumulative += useSupportSurfaceWeights ? Mathf.Max(0f, surface.selectionWeight) : 1f;

            if (randomValue <= cumulative)
            {
                return surface;
            }
        }

        return null;
    }

    private bool IsSurfaceAllowed(
        SupportSurface surface,
        SupportSurface excludedSurface,
        WeightedSurfaceDistractorPrefab distractorPrefab
    )
    {
        if (surface == null)
        {
            return false;
        }

        if (surface == excludedSurface)
        {
            return false;
        }

        return distractorPrefab.CanSpawnOn(surface);
    }

    private void IncrementPrefabCount(GameObject prefab)
    {
        if (!placedPrefabCounts.ContainsKey(prefab))
        {
            placedPrefabCounts[prefab] = 0;
        }

        placedPrefabCounts[prefab]++;
    }
}
