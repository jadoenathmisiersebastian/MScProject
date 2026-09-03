using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class WeightedWallFeaturePrefab
{
    public GameObject prefab;
    public float weight = 1f;
    public int maxInstancesPerScene = 99;

    [Header("Prefab Alignment")]
    public Vector3 rotationOffsetEuler = Vector3.zero;

    [Header("Allowed Wall Zone Types")]
    public bool allowWindow = true;
    public bool allowWallCabinet = true;
    public bool allowWallShelf = true;
    public bool allowDecoration = true;
    public bool allowLighting = true;
    public bool allowGeneral = true;

    public bool AllowsZone(WallPlacementZone zone)
    {
        if (zone == null)
        {
            return false;
        }

        switch (zone.zoneType)
        {
            case WallPlacementZoneType.Window:
                return allowWindow;
            case WallPlacementZoneType.WallCabinet:
                return allowWallCabinet;
            case WallPlacementZoneType.WallShelf:
                return allowWallShelf;
            case WallPlacementZoneType.Decoration:
                return allowDecoration;
            case WallPlacementZoneType.Lighting:
                return allowLighting;
            case WallPlacementZoneType.General:
                return allowGeneral;
            default:
                return false;
        }
    }
}

[System.Serializable]
public class WeightedWallPlacementZone
{
    public WallPlacementZone zone;
    public float weight = 1f;
}

public class WallFeatureRandomizer : MonoBehaviour
{
    [Header("Parents")]
    public Transform wallMountedFeaturesParent;

    [Header("Wall Feature Prefabs")]
    public WeightedWallFeaturePrefab[] wallFeaturePrefabs;

    [Header("Wall Placement Zones")]
    public WeightedWallPlacementZone[] wallPlacementZones;

    [Header("Generation")]
    public int minFeatureCount = 1;
    public int maxFeatureCount = 4;
    public int maxAttemptsPerFeature = 50;

    private readonly List<WallMountedFootprint> placedFootprints =
        new List<WallMountedFootprint>();

    private readonly List<GameObject> spawnedFeatures =
        new List<GameObject>();

    private readonly Dictionary<GameObject, int> placedPrefabCounts =
        new Dictionary<GameObject, int>();

    public void RandomizeWallFeatures()
    {
        ClearWallFeatures();

        int featureCount = Random.Range(minFeatureCount, maxFeatureCount + 1);

        for (int i = 0; i < featureCount; i++)
        {
            bool placed = TryPlaceWallFeature(i);

            if (!placed)
            {
                Debug.LogWarning(
                    $"Could not place wall feature {i:00} after " +
                    $"{maxAttemptsPerFeature} attempts."
                );
            }
        }
    }

    public void ClearWallFeatures()
    {
        foreach (GameObject feature in spawnedFeatures)
        {
            if (feature != null)
            {
                feature.SetActive(false);
                Destroy(feature);
            }
        }

        spawnedFeatures.Clear();
        placedFootprints.Clear();
        placedPrefabCounts.Clear();
    }

    private bool TryPlaceWallFeature(int index)
    {
        for (int attempt = 0; attempt < maxAttemptsPerFeature; attempt++)
        {
            GameObject prefab = PickWallFeaturePrefab();

            if (prefab == null)
            {
                return false;
            }

            WallPlacementZone zone = PickWallPlacementZone(prefab);

            if (zone == null)
            {
                return false;
            }

            WeightedWallFeaturePrefab featureEntry = FindWallFeatureEntry(prefab);

            GameObject candidate = Instantiate(
                prefab,
                zone.transform.position,
                SampleWallFeatureRotation(zone, featureEntry),
                wallMountedFeaturesParent
            );

            candidate.name = $"WallFeature_{index:00}_{prefab.name}";
            candidate.SetActive(false);

            WallMountedFootprint footprint =
                candidate.GetComponentInChildren<WallMountedFootprint>(true);

            if (footprint == null)
            {
                Debug.LogWarning(
                    $"Wall feature prefab {prefab.name} has no WallMountedFootprint " +
                    "on its root or children."
                );

                Destroy(candidate);
                return false;
            }

            candidate.transform.position = zone.SampleWorldPosition();
            candidate.transform.rotation = SampleWallFeatureRotation(zone, featureEntry);

            if (IsFootprintValid(footprint, zone))
            {
                candidate.SetActive(true);
                spawnedFeatures.Add(candidate);
                placedFootprints.Add(footprint);
                RegisterPlacedPrefab(prefab);
                return true;
            }

            Destroy(candidate);
        }

        return false;
    }


    private Quaternion SampleWallFeatureRotation(
        WallPlacementZone zone,
        WeightedWallFeaturePrefab featureEntry
    )
    {
        if (zone == null)
        {
            return Quaternion.identity;
        }

        Quaternion offset = Quaternion.identity;

        if (featureEntry != null)
        {
            offset = Quaternion.Euler(featureEntry.rotationOffsetEuler);
        }

        return zone.SampleWorldRotation() * offset;
    }

    private GameObject PickWallFeaturePrefab()
    {
        float totalWeight = 0f;

        foreach (WeightedWallFeaturePrefab item in wallFeaturePrefabs)
        {
            if (IsWallFeaturePrefabAvailable(item))
            {
                totalWeight += item.weight;
            }
        }

        if (totalWeight <= 0f)
        {
            Debug.LogError("No valid wall feature prefab weights.");
            return null;
        }

        float randomValue = Random.Range(0f, totalWeight);
        float cumulative = 0f;

        foreach (WeightedWallFeaturePrefab item in wallFeaturePrefabs)
        {
            if (!IsWallFeaturePrefabAvailable(item))
            {
                continue;
            }

            cumulative += item.weight;

            if (randomValue <= cumulative)
            {
                return item.prefab;
            }
        }

        return null;
    }


    private bool IsWallFeaturePrefabAvailable(WeightedWallFeaturePrefab item)
    {
        if (item == null || item.prefab == null || item.weight <= 0f)
        {
            return false;
        }

        if (item.maxInstancesPerScene < 0)
        {
            return true;
        }

        int currentCount = GetPlacedPrefabCount(item.prefab);
        return currentCount < item.maxInstancesPerScene;
    }

    private int GetPlacedPrefabCount(GameObject prefab)
    {
        if (prefab == null)
        {
            return 0;
        }

        if (placedPrefabCounts.TryGetValue(prefab, out int count))
        {
            return count;
        }

        return 0;
    }

    private void RegisterPlacedPrefab(GameObject prefab)
    {
        if (prefab == null)
        {
            return;
        }

        placedPrefabCounts[prefab] = GetPlacedPrefabCount(prefab) + 1;
    }

    private WallPlacementZone PickWallPlacementZone(GameObject prefab)
    {
        WeightedWallFeaturePrefab featureEntry = FindWallFeatureEntry(prefab);
        float totalWeight = 0f;

        foreach (WeightedWallPlacementZone item in wallPlacementZones)
        {
            if (!IsZoneAllowedForFeature(item, featureEntry))
            {
                continue;
            }

            totalWeight += item.weight;
        }

        if (totalWeight <= 0f)
        {
            Debug.LogWarning(
                $"No allowed wall placement zones found for wall feature prefab {prefab.name}."
            );

            return null;
        }

        float randomValue = Random.Range(0f, totalWeight);
        float cumulative = 0f;

        foreach (WeightedWallPlacementZone item in wallPlacementZones)
        {
            if (!IsZoneAllowedForFeature(item, featureEntry))
            {
                continue;
            }

            cumulative += item.weight;

            if (randomValue <= cumulative)
            {
                return item.zone;
            }
        }

        return null;
    }

    private WeightedWallFeaturePrefab FindWallFeatureEntry(GameObject prefab)
    {
        foreach (WeightedWallFeaturePrefab item in wallFeaturePrefabs)
        {
            if (item.prefab == prefab)
            {
                return item;
            }
        }

        return null;
    }

    private bool IsZoneAllowedForFeature(
        WeightedWallPlacementZone zoneEntry,
        WeightedWallFeaturePrefab featureEntry
    )
    {
        if (zoneEntry == null || zoneEntry.zone == null || zoneEntry.weight <= 0f)
        {
            return false;
        }

        if (featureEntry == null)
        {
            return true;
        }

        return featureEntry.AllowsZone(zoneEntry.zone);
    }

    private bool IsFootprintValid(WallMountedFootprint candidate, WallPlacementZone zone)
    {
        if (zone == null || !zone.ContainsFootprint(candidate))
        {
            return false;
        }

        foreach (WallMountedFootprint placedFootprint in placedFootprints)
        {
            if (placedFootprint == null)
            {
                continue;
            }

            if (WallFootprintsOverlap(candidate, placedFootprint))
            {
                return false;
            }
        }

        return true;
    }

    private bool WallFootprintsOverlap(
        WallMountedFootprint a,
        WallMountedFootprint b
    )
    {
        Vector3[] aCorners = a.GetWorldCorners();
        Vector3[] bCorners = b.GetWorldCorners();

        Vector2[] aPoints = ToLocalWallPoints(a.transform, aCorners);
        Vector2[] bPoints = ToLocalWallPoints(a.transform, bCorners);

        return PolygonsOverlap(aPoints, bPoints);
    }

    private Vector2[] ToLocalWallPoints(Transform reference, Vector3[] corners)
    {
        Vector2[] points = new Vector2[corners.Length];

        for (int i = 0; i < corners.Length; i++)
        {
            Vector3 localPoint = reference.InverseTransformPoint(corners[i]);
            points[i] = new Vector2(localPoint.x, localPoint.y);
        }

        return points;
    }

    private bool PolygonsOverlap(Vector2[] a, Vector2[] b)
    {
        return !HasSeparatingAxis(a, b) &&
               !HasSeparatingAxis(b, a);
    }

    private bool HasSeparatingAxis(Vector2[] a, Vector2[] b)
    {
        for (int i = 0; i < a.Length; i++)
        {
            Vector2 p1 = a[i];
            Vector2 p2 = a[(i + 1) % a.Length];

            Vector2 edge = p2 - p1;
            Vector2 axis = new Vector2(-edge.y, edge.x).normalized;

            ProjectPolygon(a, axis, out float minA, out float maxA);
            ProjectPolygon(b, axis, out float minB, out float maxB);

            if (maxA < minB || maxB < minA)
            {
                return true;
            }
        }

        return false;
    }

    private void ProjectPolygon(
        Vector2[] polygon,
        Vector2 axis,
        out float min,
        out float max
    )
    {
        min = Vector2.Dot(polygon[0], axis);
        max = min;

        for (int i = 1; i < polygon.Length; i++)
        {
            float projection = Vector2.Dot(polygon[i], axis);

            if (projection < min)
            {
                min = projection;
            }

            if (projection > max)
            {
                max = projection;
            }
        }
    }
}
