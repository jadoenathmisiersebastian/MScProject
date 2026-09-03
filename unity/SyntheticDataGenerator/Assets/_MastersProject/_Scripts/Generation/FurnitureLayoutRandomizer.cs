using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class WeightedFurniturePrefab
{
    public GameObject prefab;
    public float weight = 1f;
    public int maxInstancesPerScene = 99;

    [Header("Allowed Zone Types")]
    public bool allowOpenFloor = true;
    public bool allowWall = true;
    public bool allowCorner = true;
    public bool allowNearSupportSurface = true;

    public bool AllowsZone(PlacementZone zone)
    {
        if (zone == null)
        {
            return false;
        }

        switch (zone.zoneType)
        {
            case PlacementZoneType.OpenFloor:
                return allowOpenFloor;
            case PlacementZoneType.Wall:
                return allowWall;
            case PlacementZoneType.Corner:
                return allowCorner;
            case PlacementZoneType.NearSupportSurface:
                return allowNearSupportSurface;
            default:
                return false;
        }
    }
}

[System.Serializable]
public class WeightedPlacementZone
{
    public PlacementZone zone;
    public float weight = 1f;
}

public class FurnitureLayoutRandomizer : MonoBehaviour
{
    [Header("Parents")]
    public Transform proceduralFurnitureParent;

    [Header("Furniture Prefabs")]
    public WeightedFurniturePrefab[] furniturePrefabs;

    [Header("Placement Zones")]
    public WeightedPlacementZone[] placementZones;

    [Header("Protected Areas")]
    public FurnitureFootprint[] protectedFootprints;

    [Header("Generation")]
    public int minFurnitureCount = 2;
    public int maxFurnitureCount = 5;
    public int maxAttemptsPerFurniture = 50;

    [Header("Rotation Control")]
    public bool alignWallPreferredFurnitureToZone = true;
    public bool useRightAngleRotationsForOpenFloor = true;
    public Vector2 wallAlignedYawJitter = new Vector2(-4f, 4f);
    public Vector2 rightAngleYawJitter = new Vector2(-8f, 8f);

    private readonly List<FurnitureFootprint> placedFootprints =
        new List<FurnitureFootprint>();

    private readonly List<GameObject> spawnedFurniture =
        new List<GameObject>();

    private readonly Dictionary<GameObject, int> placedPrefabCounts =
        new Dictionary<GameObject, int>();

    public void RandomizeFurnitureLayout()
    {
        ClearFurniture();

        int furnitureCount = Random.Range(
            minFurnitureCount,
            maxFurnitureCount + 1
        );

        for (int i = 0; i < furnitureCount; i++)
        {
            bool placed = TryPlaceFurniture(i);

            if (!placed)
            {
                Debug.LogWarning(
                    $"Could not place furniture item {i:00} after " +
                    $"{maxAttemptsPerFurniture} attempts."
                );
            }
        }
    }

    public void ClearFurniture()
    {
        foreach (GameObject furniture in spawnedFurniture)
        {
            if (furniture != null)
            {
                furniture.SetActive(false);
                Destroy(furniture);
            }
        }

        spawnedFurniture.Clear();
        placedFootprints.Clear();
        placedPrefabCounts.Clear();
    }

    private bool TryPlaceFurniture(int index)
    {
        for (int attempt = 0; attempt < maxAttemptsPerFurniture; attempt++)
        {
            GameObject prefab = PickFurniturePrefab();

            if (prefab == null)
            {
                return false;
            }

            PlacementZone zone = PickPlacementZone(prefab);

            if (zone == null)
            {
                return false;
            }

            GameObject candidate = Instantiate(
                prefab,
                zone.transform.position,
                Quaternion.identity,
                proceduralFurnitureParent
            );

            candidate.name = $"Furniture_{index:00}_{prefab.name}";
            candidate.SetActive(false);

            FurnitureFootprint footprint =
                candidate.GetComponentInChildren<FurnitureFootprint>(true);

            if (footprint == null)
            {
                Debug.LogWarning(
                    $"Furniture prefab {prefab.name} has no FurnitureFootprint " +
                    "on its root or children. Add one to a FootprintRoot child."
                );

                Destroy(candidate);
                return false;
            }

            candidate.transform.rotation = SampleFurnitureRotation(zone, footprint);
            candidate.transform.position = zone.SampleWorldPosition();

            if (IsFootprintValid(footprint, zone))
            {
                candidate.SetActive(true);
                spawnedFurniture.Add(candidate);
                placedFootprints.Add(footprint);
                RegisterPlacedPrefab(prefab);
                return true;
            }

            Destroy(candidate);
        }

        return false;
    }


    private Quaternion SampleFurnitureRotation(PlacementZone zone, FurnitureFootprint footprint)
    {
        if (zone == null)
        {
            return Quaternion.identity;
        }

        if (footprint != null && footprint.preferAgainstWall && alignWallPreferredFurnitureToZone)
        {
            float yaw = zone.transform.eulerAngles.y + Random.Range(
                wallAlignedYawJitter.x,
                wallAlignedYawJitter.y
            );

            return Quaternion.Euler(0f, yaw, 0f);
        }

        if (zone.zoneType == PlacementZoneType.Wall && alignWallPreferredFurnitureToZone)
        {
            float yaw = zone.transform.eulerAngles.y + Random.Range(
                wallAlignedYawJitter.x,
                wallAlignedYawJitter.y
            );

            return Quaternion.Euler(0f, yaw, 0f);
        }

        if (useRightAngleRotationsForOpenFloor && zone.zoneType == PlacementZoneType.OpenFloor)
        {
            int rightAngleStep = Random.Range(0, 4);
            float yaw = rightAngleStep * 90f + Random.Range(
                rightAngleYawJitter.x,
                rightAngleYawJitter.y
            );

            return Quaternion.Euler(0f, yaw, 0f);
        }

        return zone.SampleWorldRotation();
    }

    private GameObject PickFurniturePrefab()
    {
        float totalWeight = 0f;

        foreach (WeightedFurniturePrefab item in furniturePrefabs)
        {
            if (IsFurniturePrefabAvailable(item))
            {
                totalWeight += item.weight;
            }
        }

        if (totalWeight <= 0f)
        {
            Debug.LogError("No valid furniture prefab weights.");
            return null;
        }

        float randomValue = Random.Range(0f, totalWeight);
        float cumulative = 0f;

        foreach (WeightedFurniturePrefab item in furniturePrefabs)
        {
            if (!IsFurniturePrefabAvailable(item))
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


    private bool IsFurniturePrefabAvailable(WeightedFurniturePrefab item)
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

    private PlacementZone PickPlacementZone(GameObject prefab)
    {
        WeightedFurniturePrefab furnitureEntry = FindFurnitureEntry(prefab);

        float totalWeight = 0f;

        foreach (WeightedPlacementZone item in placementZones)
        {
            if (!IsZoneAllowedForFurniture(item, furnitureEntry))
            {
                continue;
            }

            totalWeight += item.weight;
        }

        if (totalWeight <= 0f)
        {
            Debug.LogWarning(
                $"No allowed placement zones found for furniture prefab {prefab.name}."
            );

            return null;
        }

        float randomValue = Random.Range(0f, totalWeight);
        float cumulative = 0f;

        foreach (WeightedPlacementZone item in placementZones)
        {
            if (!IsZoneAllowedForFurniture(item, furnitureEntry))
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

    private WeightedFurniturePrefab FindFurnitureEntry(GameObject prefab)
    {
        foreach (WeightedFurniturePrefab item in furniturePrefabs)
        {
            if (item.prefab == prefab)
            {
                return item;
            }
        }

        return null;
    }

    private bool IsZoneAllowedForFurniture(
        WeightedPlacementZone zoneEntry,
        WeightedFurniturePrefab furnitureEntry
    )
    {
        if (zoneEntry == null || zoneEntry.zone == null || zoneEntry.weight <= 0f)
        {
            return false;
        }

        if (furnitureEntry == null)
        {
            return true;
        }

        return furnitureEntry.AllowsZone(zoneEntry.zone);
    }

    private bool IsFootprintValid(FurnitureFootprint candidate, PlacementZone zone)
    {
        if (zone == null || !zone.ContainsFootprint(candidate))
        {
            return false;
        }

        foreach (FurnitureFootprint protectedFootprint in protectedFootprints)
        {
            if (protectedFootprint == null)
            {
                continue;
            }

            if (FootprintsOverlap(candidate, protectedFootprint))
            {
                return false;
            }
        }

        foreach (FurnitureFootprint placedFootprint in placedFootprints)
        {
            if (placedFootprint == null)
            {
                continue;
            }

            if (FootprintsOverlap(candidate, placedFootprint))
            {
                return false;
            }
        }

        return true;
    }

    private bool FootprintsOverlap(
        FurnitureFootprint a,
        FurnitureFootprint b
    )
    {
        Vector3[] aCorners = a.GetWorldCorners();
        Vector3[] bCorners = b.GetWorldCorners();

        Vector2[] aPoints = ToXZPoints(aCorners);
        Vector2[] bPoints = ToXZPoints(bCorners);

        return PolygonsOverlap(aPoints, bPoints);
    }

    private Vector2[] ToXZPoints(Vector3[] corners)
    {
        Vector2[] points = new Vector2[corners.Length];

        for (int i = 0; i < corners.Length; i++)
        {
            points[i] = new Vector2(corners[i].x, corners[i].z);
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