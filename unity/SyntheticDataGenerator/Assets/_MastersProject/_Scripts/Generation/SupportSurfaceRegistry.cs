using System.Collections.Generic;
using UnityEngine;

public class SupportSurfaceRegistry : MonoBehaviour
{
    [Header("Roots")]
    public Transform[] searchRoots;

    [Header("Debug")]
    public bool logRefreshSummary = false;

    private readonly List<SupportSurface> availableSurfaces =
        new List<SupportSurface>();

    public IReadOnlyList<SupportSurface> AvailableSurfaces => availableSurfaces;

    public void Refresh()
    {
        availableSurfaces.Clear();
        HashSet<SupportSurface> seenSurfaces = new HashSet<SupportSurface>();

        foreach (Transform root in searchRoots)
        {
            if (root == null)
            {
                continue;
            }

            SupportSurface[] surfaces =
                root.GetComponentsInChildren<SupportSurface>(false);

            foreach (SupportSurface surface in surfaces)
            {
                if (surface == null)
                {
                    continue;
                }

                if (!surface.gameObject.activeInHierarchy)
                {
                    continue;
                }

                if (!surface.allowTargetSpawning)
                {
                    continue;
                }

                if (surface.selectionWeight <= 0f)
                {
                    continue;
                }

                if (!seenSurfaces.Add(surface))
                {
                    continue;
                }

                availableSurfaces.Add(surface);
            }
        }

        if (logRefreshSummary)
        {
            Debug.Log(
                $"Support surface registry found " +
                $"{availableSurfaces.Count} available surfaces:\n" +
                BuildSurfaceSummary()
            );
        }
    }

    private string BuildSurfaceSummary()
    {
        if (availableSurfaces.Count == 0)
        {
            return "<none>";
        }

        List<string> lines = new List<string>();

        foreach (SupportSurface surface in availableSurfaces)
        {
            lines.Add(
                $"- {surface.surfaceId} | " +
                $"weight={surface.selectionWeight:F2} | " +
                GetTransformPath(surface.transform)
            );
        }

        return string.Join("\n", lines);
    }

    private string GetTransformPath(Transform target)
    {
        if (target == null)
        {
            return "<null>";
        }

        List<string> names = new List<string>();
        Transform current = target;

        while (current != null)
        {
            names.Add(current.name);
            current = current.parent;
        }

        names.Reverse();
        return string.Join("/", names);
    }

    public SupportSurface GetRandomSurface()
    {
        Refresh();

        if (availableSurfaces.Count == 0)
        {
            Debug.LogError("No available support surfaces found.");
            return null;
        }

        float totalWeight = 0f;

        foreach (SupportSurface surface in availableSurfaces)
        {
            totalWeight += surface.selectionWeight;
        }

        float randomValue = Random.Range(0f, totalWeight);
        float cumulative = 0f;

        foreach (SupportSurface surface in availableSurfaces)
        {
            cumulative += surface.selectionWeight;

            if (randomValue <= cumulative)
            {
                return surface;
            }
        }

        return availableSurfaces[availableSurfaces.Count - 1];
    }
}