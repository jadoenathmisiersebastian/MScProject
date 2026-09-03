using System;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

[Serializable]
[AddRandomizerMenu("Custom/Background Placement Randomizer")]
public class NewPlacementRandomizer : Randomizer
{
    public Transform[] spawnCenters;

    public Vector3Parameter positionDistribution = new Vector3Parameter
    {
        x = new UniformSampler(-2f, 2f),
        y = new UniformSampler(0f, 2f),
        z = new UniformSampler(0f, 2f)

    };

    public IntegerParameter objectCount = new IntegerParameter
    {
        value = new ConstantSampler(10)
    };

    public CategoricalParameter<GameObject> prefabs;

    GameObject m_Container;

    protected override void OnAwake()
    {
        m_Container = new GameObject("BackgroundObjects");
        m_Container.transform.parent = scenario.transform;
    }

    protected override void OnIterationStart()
    {
        if (spawnCenters == null || spawnCenters.Length == 0)
        {
            Debug.LogError("No spawn centers assigned!");
            return;
        }

        var count = objectCount.Sample();

        for (int i = 0; i < count; i++)
        {
            var prefab = prefabs.Sample();
            var instance = GameObject.Instantiate(prefab, m_Container.transform);

            var center = spawnCenters[UnityEngine.Random.Range(0, spawnCenters.Length)];

            Vector3 localPos = positionDistribution.Sample();

            // Apply placement in the selected spawn centre's local frame.
            instance.transform.position = center.TransformPoint(localPos);

            instance.transform.rotation =
                center.rotation * Quaternion.Euler(0, UnityEngine.Random.Range(0, 360), 0);
        }
    }

    protected override void OnIterationEnd()
    {
        foreach (Transform child in m_Container.transform)
        {
            GameObject.Destroy(child.gameObject);
        }
    }
}
