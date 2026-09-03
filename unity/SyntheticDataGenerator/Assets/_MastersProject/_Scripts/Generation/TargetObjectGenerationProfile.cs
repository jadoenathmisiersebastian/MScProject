using UnityEngine;

[DisallowMultipleComponent]
public class TargetObjectGenerationProfile : MonoBehaviour
{
    [Header("Discrete Resting Pose Weights")]
    [Min(0f)] public float uprightWeight = 10f;
    [Min(0f)] public float upsideDownWeight = 1f;
    [Min(0f)] public float sideZPositiveWeight = 1f;
    [Min(0f)] public float sideZNegativeWeight = 1f;
    [Min(0f)] public float sideXPositiveWeight = 1f;
    [Min(0f)] public float sideXNegativeWeight = 1f;

    public float[] GetRestingPoseWeights()
    {
        return new float[]
        {
            Mathf.Max(0f, uprightWeight),
            Mathf.Max(0f, upsideDownWeight),
            Mathf.Max(0f, sideZPositiveWeight),
            Mathf.Max(0f, sideZNegativeWeight),
            Mathf.Max(0f, sideXPositiveWeight),
            Mathf.Max(0f, sideXNegativeWeight)
        };
    }

    private void OnValidate()
    {
        uprightWeight = Mathf.Max(0f, uprightWeight);
        upsideDownWeight = Mathf.Max(0f, upsideDownWeight);
        sideZPositiveWeight = Mathf.Max(0f, sideZPositiveWeight);
        sideZNegativeWeight = Mathf.Max(0f, sideZNegativeWeight);
        sideXPositiveWeight = Mathf.Max(0f, sideXPositiveWeight);
        sideXNegativeWeight = Mathf.Max(0f, sideXNegativeWeight);
    }
}
