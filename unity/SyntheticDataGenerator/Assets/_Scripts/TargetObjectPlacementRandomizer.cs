using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

[Serializable]
[AddRandomizerMenu("Custom/Foreground Placement Randomizer")]
public class ForegroundPlacementRandomizer : Randomizer
{
    public Transform table;

    public Vector2 tableSize = new Vector2(2f, 2f);

    public float spawnHeight = 2f;

    public IntegerParameter objectCount = new IntegerParameter
    {
        value = new ConstantSampler(5)
    };

    public CategoricalParameter<GameObject> prefabs;

    GameObject m_Container;

    protected override void OnAwake()
    {
        m_Container = new GameObject("ForegroundObjects");
        m_Container.transform.parent = scenario.transform;
    }

    protected override void OnIterationStart()
    {
        var count = objectCount.Sample();

        for (int i = 0; i < count; i++)
        {
            var prefab = prefabs.Sample();

            var instance = GameObject.Instantiate(prefab, m_Container.transform);

            float x = UnityEngine.Random.Range(-tableSize.x / 2, tableSize.x / 2);
            float z = UnityEngine.Random.Range(-tableSize.y / 2, tableSize.y / 2);

            Vector3 spawnPos = table.TransformPoint(new Vector3(x, spawnHeight, z));

            instance.transform.position = spawnPos;

            // Restrict rotation to stable resting faces, then vary yaw.
            instance.transform.rotation = GetRandomSpawnRotation();
        }

        scenario.StartCoroutine(WaitForPhysics());
    }

    IEnumerator WaitForPhysics()
    {
        yield return new WaitForSeconds(1.5f);
    }

    Quaternion GetRandomSpawnRotation()
    {
        Quaternion[] restingRotations =
        {
            Quaternion.Euler(0f, 0f, 0f),
            Quaternion.Euler(180f, 0f, 0f),
            Quaternion.Euler(0f, 0f, 90f),
            Quaternion.Euler(0f, 0f, -90f),
            Quaternion.Euler(90f, 0f, 0f),
            Quaternion.Euler(-90f, 0f, 0f),
        };

        Quaternion restingRotation = restingRotations[UnityEngine.Random.Range(0, restingRotations.Length)];
        Quaternion yawRotation = Quaternion.Euler(0f, UnityEngine.Random.Range(0f, 360f), 0f);

        return yawRotation * restingRotation;
    }


    protected override void OnIterationEnd()
    {
        foreach (Transform child in m_Container.transform)
        {
            GameObject.Destroy(child.gameObject);
        }
    }
}
