using UnityEngine;

public enum WallFeatureCategory
{
    Window,
    WallCabinet,
    WallShelf,
    Painting,
    WallLamp,
    WallRack,
    Other
}

public class WallMountedFootprint : MonoBehaviour
{
    [Header("Category")]
    public WallFeatureCategory category = WallFeatureCategory.Other;

    [Header("Footprint")]
    public Vector2 size = new Vector2(0.8f, 0.6f);
    public float padding = 0.03f;
    public float wallOffset = 0.02f;

    public Vector2 PaddedSize
    {
        get
        {
            return new Vector2(
                size.x + padding * 2f,
                size.y + padding * 2f
            );
        }
    }

    public Vector3[] GetWorldCorners()
    {
        Vector2 paddedSize = PaddedSize;

        Vector3[] localCorners =
        {
            new Vector3(-paddedSize.x * 0.5f, -paddedSize.y * 0.5f, wallOffset),
            new Vector3( paddedSize.x * 0.5f, -paddedSize.y * 0.5f, wallOffset),
            new Vector3( paddedSize.x * 0.5f,  paddedSize.y * 0.5f, wallOffset),
            new Vector3(-paddedSize.x * 0.5f,  paddedSize.y * 0.5f, wallOffset)
        };

        Vector3[] worldCorners = new Vector3[4];

        for (int i = 0; i < localCorners.Length; i++)
        {
            worldCorners[i] = transform.TransformPoint(localCorners[i]);
        }

        return worldCorners;
    }

    private void OnDrawGizmos()
    {
        Vector2 paddedSize = PaddedSize;

        Gizmos.color = new Color(0.6f, 0.2f, 1f, 0.25f);

        Matrix4x4 oldMatrix = Gizmos.matrix;
        Gizmos.matrix = transform.localToWorldMatrix;

        Gizmos.DrawCube(
            new Vector3(0f, 0f, wallOffset),
            new Vector3(paddedSize.x, paddedSize.y, 0.025f)
        );

        Gizmos.color = new Color(0.6f, 0.2f, 1f, 0.9f);
        Gizmos.DrawWireCube(
            new Vector3(0f, 0f, wallOffset),
            new Vector3(paddedSize.x, paddedSize.y, 0.025f)
        );

        Gizmos.matrix = oldMatrix;
    }
}
