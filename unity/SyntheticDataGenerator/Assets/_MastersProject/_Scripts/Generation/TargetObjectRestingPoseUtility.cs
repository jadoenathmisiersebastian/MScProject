using UnityEngine;

public enum TargetObjectSpawnMode
{
    DropFromHeight,
    PlaceOnSurface
}

public static class TargetObjectRestingPoseUtility
{
    private static readonly Quaternion[] RestingRotations =
    {
        Quaternion.Euler(0f, 0f, 0f),
        Quaternion.Euler(180f, 0f, 0f),
        Quaternion.Euler(0f, 0f, 90f),
        Quaternion.Euler(0f, 0f, -90f),
        Quaternion.Euler(90f, 0f, 0f),
        Quaternion.Euler(-90f, 0f, 0f)
    };

    public static Quaternion GetRandomDiscreteRestingPoseWithYaw(float[] restingPoseWeights = null)
    {
        Quaternion restingRotation = GetRandomRestingRotation(restingPoseWeights);

        Quaternion yawRotation = Quaternion.Euler(
            0f,
            Random.Range(0f, 360f),
            0f
        );

        return yawRotation * restingRotation;
    }

    public static Quaternion GetRandomYawOnlyPose()
    {
        return Quaternion.Euler(
            0f,
            Random.Range(0f, 360f),
            0f
        );
    }

    public static void AlignRendererBoundsBottomToWorldY(GameObject obj, float targetBottomY)
    {
        if (obj == null)
        {
            return;
        }

        if (!TryGetRendererBounds(obj, out Bounds bounds))
        {
            return;
        }

        Vector3 position = obj.transform.position;
        position.y += targetBottomY - bounds.min.y;
        obj.transform.position = position;
    }

    public static bool TryGetRendererBounds(GameObject obj, out Bounds bounds)
    {
        bounds = default;

        if (obj == null)
        {
            return false;
        }

        Renderer[] renderers = obj.GetComponentsInChildren<Renderer>();

        if (renderers.Length == 0)
        {
            return false;
        }

        bounds = renderers[0].bounds;

        for (int i = 1; i < renderers.Length; i++)
        {
            bounds.Encapsulate(renderers[i].bounds);
        }

        return true;
    }

    public static void ResetRigidbodyVelocities(GameObject obj)
    {
        if (obj == null)
        {
            return;
        }

        Rigidbody[] rigidbodies = obj.GetComponentsInChildren<Rigidbody>();

        foreach (Rigidbody rb in rigidbodies)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            rb.WakeUp();
        }
    }

    private static Quaternion GetRandomRestingRotation(float[] weights)
    {
        if (weights == null || weights.Length == 0)
        {
            return RestingRotations[Random.Range(0, RestingRotations.Length)];
        }

        float totalWeight = 0f;

        for (int i = 0; i < RestingRotations.Length; i++)
        {
            float weight = i < weights.Length ? Mathf.Max(0f, weights[i]) : 0f;
            totalWeight += weight;
        }

        if (totalWeight <= 0f)
        {
            return RestingRotations[Random.Range(0, RestingRotations.Length)];
        }

        float randomValue = Random.Range(0f, totalWeight);
        float cumulative = 0f;

        for (int i = 0; i < RestingRotations.Length; i++)
        {
            float weight = i < weights.Length ? Mathf.Max(0f, weights[i]) : 0f;
            cumulative += weight;

            if (randomValue <= cumulative)
            {
                return RestingRotations[i];
            }
        }

        return RestingRotations[RestingRotations.Length - 1];
    }
}
