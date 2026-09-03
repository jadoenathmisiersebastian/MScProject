using UnityEngine;

public class EnvironmentLayoutRandomizer : MonoBehaviour
{
    [Header("Table Environment")]
    public Transform tableEnvironmentRoot;

    [Header("Position Offset Range")]
    public Vector2 xOffsetRange = new Vector2(-0.25f, 0.25f);
    public Vector2 zOffsetRange = new Vector2(-0.15f, 0.15f);

    [Header("Rotation Range")]
    public Vector2 yawOffsetRange = new Vector2(-8f, 8f);

    [Header("Options")]
    public bool resetToInitialPoseBeforeRandomizing = true;

    private Vector3 initialPosition;
    private Quaternion initialRotation;
    private bool hasInitialPose;

    private void Awake()
    {
        CacheInitialPose();
    }

    public void RandomizeEnvironment()
    {
        if (tableEnvironmentRoot == null)
        {
            Debug.LogError("Environment layout randomizer table environment root is not assigned.");
            return;
        }

        CacheInitialPose();

        if (resetToInitialPoseBeforeRandomizing)
        {
            ResetEnvironment();
        }

        Vector3 offset = new Vector3(
            Random.Range(xOffsetRange.x, xOffsetRange.y),
            0f,
            Random.Range(zOffsetRange.x, zOffsetRange.y)
        );

        float yawOffset = Random.Range(yawOffsetRange.x, yawOffsetRange.y);

        tableEnvironmentRoot.position = initialPosition + offset;
        tableEnvironmentRoot.rotation =
            initialRotation * Quaternion.Euler(0f, yawOffset, 0f);
    }

    public void ResetEnvironment()
    {
        if (tableEnvironmentRoot == null)
        {
            return;
        }

        CacheInitialPose();

        tableEnvironmentRoot.position = initialPosition;
        tableEnvironmentRoot.rotation = initialRotation;
    }

    private void CacheInitialPose()
    {
        if (hasInitialPose || tableEnvironmentRoot == null)
        {
            return;
        }

        initialPosition = tableEnvironmentRoot.position;
        initialRotation = tableEnvironmentRoot.rotation;
        hasInitialPose = true;
    }
}