using UnityEngine;

public enum WallPlacementZoneType
{
    Window,
    WallCabinet,
    WallShelf,
    Decoration,
    Lighting,
    General
}

public class WallPlacementZone : MonoBehaviour
{
    [Header("Zone")]
    public WallPlacementZoneType zoneType = WallPlacementZoneType.General;

    [Header("Local Bounds")]
    public Vector2 size = new Vector2(1.2f, 0.8f);

    public Vector3 SampleWorldPosition()
    {
        float localX = Random.Range(-size.x * 0.5f, size.x * 0.5f);
        float localY = Random.Range(-size.y * 0.5f, size.y * 0.5f);

        return transform.TransformPoint(new Vector3(localX, localY, 0f));
    }

    public Quaternion SampleWorldRotation()
    {
        return transform.rotation;
    }

    public bool ContainsWorldPoint(Vector3 worldPoint)
    {
        Vector3 localPoint = transform.InverseTransformPoint(worldPoint);

        return Mathf.Abs(localPoint.x) <= size.x * 0.5f &&
               Mathf.Abs(localPoint.y) <= size.y * 0.5f;
    }

    public bool ContainsFootprint(WallMountedFootprint footprint)
    {
        if (footprint == null)
        {
            return false;
        }

        Vector3[] corners = footprint.GetWorldCorners();

        foreach (Vector3 corner in corners)
        {
            if (!ContainsWorldPoint(corner))
            {
                return false;
            }
        }

        return true;
    }

    private void OnDrawGizmos()
    {
        Color color = Color.white;

        switch (zoneType)
        {
            case WallPlacementZoneType.Window:
                color = Color.cyan;
                break;
            case WallPlacementZoneType.WallCabinet:
                color = Color.yellow;
                break;
            case WallPlacementZoneType.WallShelf:
                color = Color.magenta;
                break;
            case WallPlacementZoneType.Decoration:
                color = Color.green;
                break;
            case WallPlacementZoneType.Lighting:
                color = new Color(1f, 0.7f, 0.2f);
                break;
        }

        color.a = 0.25f;
        Gizmos.color = color;

        Matrix4x4 oldMatrix = Gizmos.matrix;
        Gizmos.matrix = transform.localToWorldMatrix;

        Gizmos.DrawCube(Vector3.zero, new Vector3(size.x, size.y, 0.025f));

        Gizmos.color = new Color(color.r, color.g, color.b, 0.9f);
        Gizmos.DrawWireCube(Vector3.zero, new Vector3(size.x, size.y, 0.025f));

        Gizmos.matrix = oldMatrix;
    }
}
