using UnityEngine;

public class SupportSurface : MonoBehaviour
{
    [Header("Identity")]
    public string surfaceId = "support_surface";

    [Header("Sampling")]
    public Vector2 spawnAreaSize = new Vector2(0.8f, 0.5f);
    public float objectSpawnHeight = 0.2f;
    public float selectionWeight = 1f;
    public bool allowTargetSpawning = true;

    [Header("Validation")]
    public float edgeMargin = 0.03f;
    public float fallBelowSurfaceThreshold = 0.08f;

    [Header("Camera Focus")]
    public Transform focusPoint;
    public Vector3 focusOffset = new Vector3(0f, 0.12f, 0f);

    public Vector3 SampleSpawnPosition()
    {
        float localX = Random.Range(
            -spawnAreaSize.x * 0.5f,
            spawnAreaSize.x * 0.5f
        );

        float localZ = Random.Range(
            -spawnAreaSize.y * 0.5f,
            spawnAreaSize.y * 0.5f
        );

        Vector3 localPosition = new Vector3(
            localX,
            objectSpawnHeight,
            localZ
        );

        return transform.TransformPoint(localPosition);
    }

    public Vector3 GetFocusPosition()
    {
        if (focusPoint != null)
        {
            return focusPoint.position;
        }

        return transform.TransformPoint(focusOffset);
    }

    public bool IsObjectOnSurface(GameObject obj)
    {
        if (obj == null)
        {
            return false;
        }

        Bounds bounds = GetObjectBounds(obj);

        if (bounds.size == Vector3.zero)
        {
            return false;
        }

        Vector3 localCenter = transform.InverseTransformPoint(bounds.center);

        if (localCenter.y < -fallBelowSurfaceThreshold)
        {
            return false;
        }

        float allowedX = spawnAreaSize.x * 0.5f + edgeMargin;
        float allowedZ = spawnAreaSize.y * 0.5f + edgeMargin;

        return Mathf.Abs(localCenter.x) <= allowedX &&
               Mathf.Abs(localCenter.z) <= allowedZ;
    }

    public Bounds GetObjectBounds(GameObject obj)
    {
        Renderer[] renderers = obj.GetComponentsInChildren<Renderer>();

        if (renderers.Length == 0)
        {
            return new Bounds(obj.transform.position, Vector3.zero);
        }

        Bounds bounds = renderers[0].bounds;

        for (int i = 1; i < renderers.Length; i++)
        {
            bounds.Encapsulate(renderers[i].bounds);
        }

        return bounds;
    }

    private void OnDrawGizmos()
    {
        Gizmos.color = new Color(0.1f, 0.8f, 1f, 0.25f);

        Matrix4x4 oldMatrix = Gizmos.matrix;
        Gizmos.matrix = transform.localToWorldMatrix;

        Gizmos.DrawCube(
            Vector3.zero,
            new Vector3(spawnAreaSize.x, 0.025f, spawnAreaSize.y)
        );

        Gizmos.color = new Color(0.1f, 0.8f, 1f, 0.9f);

        Gizmos.DrawWireCube(
            Vector3.zero,
            new Vector3(spawnAreaSize.x, 0.025f, spawnAreaSize.y)
        );

        Gizmos.matrix = oldMatrix;

        Gizmos.color = Color.blue;
        Gizmos.DrawSphere(GetFocusPosition(), 0.035f);
    }
}