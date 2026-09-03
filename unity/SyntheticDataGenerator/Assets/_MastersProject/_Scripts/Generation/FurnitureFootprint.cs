using UnityEngine;

public enum FurnitureCategory
{
    Table,
    Chair,
    Counter,
    Cabinet,
    Shelf,
    Plant,
    Bin,
    Decorative,
    Other
}

public class FurnitureFootprint : MonoBehaviour
{
    [Header("Category")]
    public FurnitureCategory category = FurnitureCategory.Other;

    [Header("Footprint")]
    public Vector2 size = new Vector2(0.5f, 0.5f);
    public float padding = 0.05f;

    [Header("Placement")]
    public bool preferAgainstWall = false;
    public bool canSupportTargetObjects = false;

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
            new Vector3(-paddedSize.x * 0.5f, 0f, -paddedSize.y * 0.5f),
            new Vector3( paddedSize.x * 0.5f, 0f, -paddedSize.y * 0.5f),
            new Vector3( paddedSize.x * 0.5f, 0f,  paddedSize.y * 0.5f),
            new Vector3(-paddedSize.x * 0.5f, 0f,  paddedSize.y * 0.5f)
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

        Gizmos.color = new Color(1f, 0.4f, 0f, 0.25f);

        Matrix4x4 oldMatrix = Gizmos.matrix;
        Gizmos.matrix = transform.localToWorldMatrix;

        Gizmos.DrawCube(
            Vector3.zero,
            new Vector3(paddedSize.x, 0.03f, paddedSize.y)
        );

        Gizmos.color = new Color(1f, 0.4f, 0f, 0.9f);
        Gizmos.DrawWireCube(
            Vector3.zero,
            new Vector3(paddedSize.x, 0.03f, paddedSize.y)
        );

        Gizmos.matrix = oldMatrix;
    }
}