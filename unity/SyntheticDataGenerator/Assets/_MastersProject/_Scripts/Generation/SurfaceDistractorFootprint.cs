using UnityEngine;

public enum SurfaceDistractorCategory
{
    Microwave,
    Toaster,
    Kettle,
    Mug,
    Book,
    Bowl,
    Decoration,
    SmallAppliance,
    Other
}

public class SurfaceDistractorFootprint : MonoBehaviour
{
    [Header("Identity")]
    public SurfaceDistractorCategory category = SurfaceDistractorCategory.Other;

    [Header("Footprint")]
    public Vector2 size = new Vector2(0.2f, 0.2f);
    public float padding = 0.02f;

    [Tooltip("Local Y offset above the support surface when this footprint root is placed.")]
    public float surfaceHeightOffset = 0.0f;

    public Vector2 PaddedSize => new Vector2(
        size.x + padding * 2f,
        size.y + padding * 2f
    );

    public Vector3[] GetWorldCorners(bool includePadding)
    {
        Vector2 footprintSize = includePadding ? PaddedSize : size;
        float halfX = footprintSize.x * 0.5f;
        float halfZ = footprintSize.y * 0.5f;

        return new Vector3[]
        {
            transform.TransformPoint(new Vector3(-halfX, 0f, -halfZ)),
            transform.TransformPoint(new Vector3( halfX, 0f, -halfZ)),
            transform.TransformPoint(new Vector3( halfX, 0f,  halfZ)),
            transform.TransformPoint(new Vector3(-halfX, 0f,  halfZ))
        };
    }

    private void OnDrawGizmos()
    {
        Matrix4x4 oldMatrix = Gizmos.matrix;
        Gizmos.matrix = transform.localToWorldMatrix;

        Gizmos.color = new Color(0.9f, 0.2f, 1f, 0.2f);
        Gizmos.DrawCube(Vector3.zero, new Vector3(size.x, 0.02f, size.y));

        Gizmos.color = new Color(0.9f, 0.2f, 1f, 0.9f);
        Gizmos.DrawWireCube(Vector3.zero, new Vector3(size.x, 0.02f, size.y));

        if (padding > 0f)
        {
            Vector2 paddedSize = PaddedSize;
            Gizmos.color = new Color(1f, 0.65f, 0.1f, 0.9f);
            Gizmos.DrawWireCube(Vector3.zero, new Vector3(paddedSize.x, 0.025f, paddedSize.y));
        }

        Gizmos.matrix = oldMatrix;
    }
}
