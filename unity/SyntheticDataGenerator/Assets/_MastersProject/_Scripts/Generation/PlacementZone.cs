using UnityEngine;

public enum PlacementZoneType
{
    OpenFloor,
    Wall,
    Corner,
    NearSupportSurface
}

public class PlacementZone : MonoBehaviour
{
    [Header("Zone")]
    public PlacementZoneType zoneType = PlacementZoneType.OpenFloor;

    [Header("Local Bounds")]
    public Vector2 size = new Vector2(1f, 1f);

    [Header("Sampling")]
    public bool allowRandomYaw = true;
    public Vector2 yawRange = new Vector2(0f, 360f);

    public Vector3 SampleWorldPosition()
    {
        float localX = Random.Range(-size.x * 0.5f, size.x * 0.5f);
        float localZ = Random.Range(-size.y * 0.5f, size.y * 0.5f);

        return transform.TransformPoint(new Vector3(localX, 0f, localZ));
    }

    public Quaternion SampleWorldRotation()
    {
        float yaw = allowRandomYaw
            ? Random.Range(yawRange.x, yawRange.y)
            : transform.eulerAngles.y;

        return Quaternion.Euler(0f, yaw, 0f);
    }

    public bool ContainsWorldPoint(Vector3 worldPoint)
    {
        Vector3 localPoint = transform.InverseTransformPoint(worldPoint);

        return Mathf.Abs(localPoint.x) <= size.x * 0.5f &&
               Mathf.Abs(localPoint.z) <= size.y * 0.5f;
    }

    public bool ContainsFootprint(FurnitureFootprint footprint)
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
        Color color = Color.green;

        switch (zoneType)
        {
            case PlacementZoneType.Wall:
                color = Color.cyan;
                break;
            case PlacementZoneType.Corner:
                color = Color.yellow;
                break;
            case PlacementZoneType.NearSupportSurface:
                color = Color.magenta;
                break;
        }

        color.a = 0.25f;
        Gizmos.color = color;

        Matrix4x4 oldMatrix = Gizmos.matrix;
        Gizmos.matrix = transform.localToWorldMatrix;

        Gizmos.DrawCube(
            Vector3.zero,
            new Vector3(size.x, 0.02f, size.y)
        );

        Gizmos.color = new Color(color.r, color.g, color.b, 0.9f);
        Gizmos.DrawWireCube(
            Vector3.zero,
            new Vector3(size.x, 0.02f, size.y)
        );

        Gizmos.matrix = oldMatrix;
    }
}