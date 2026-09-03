using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Perception.GroundTruth.LabelManagement;

public class PerceptionObjectLibrary : MonoBehaviour
{
    [Header("Target Object Prefabs")]
    public GameObject[] prefabs;

    [Header("Semantic Sampling")]
    [Tooltip("Select each semantic class equally, then select a prefab within that class.")]
    public bool balanceSemanticClasses = true;

    private readonly List<List<GameObject>> prefabClasses =
        new List<List<GameObject>>();

    private readonly List<string> semanticClasses = new List<string>();
    private readonly List<int> shuffledClassBag = new List<int>();
    private bool classCacheBuilt;

    public GameObject GetRandomPrefab()
    {
        if (prefabs == null || prefabs.Length == 0)
        {
            Debug.LogError("Perception object library has no prefabs assigned.");
            return null;
        }

        if (!balanceSemanticClasses)
        {
            return prefabs[Random.Range(0, prefabs.Length)];
        }

        if (!classCacheBuilt && !BuildClassCache())
        {
            return null;
        }

        if (shuffledClassBag.Count == 0)
        {
            RefillClassBag();
        }

        int lastIndex = shuffledClassBag.Count - 1;
        int classIndex = shuffledClassBag[lastIndex];
        shuffledClassBag.RemoveAt(lastIndex);

        List<GameObject> classPrefabs = prefabClasses[classIndex];
        return classPrefabs[Random.Range(0, classPrefabs.Count)];
    }

    public float[] GetRestingPoseWeights(
        GameObject prefab,
        float[] fallbackWeights
    )
    {
        if (prefab != null)
        {
            TargetObjectGenerationProfile profile =
                prefab.GetComponentInChildren<TargetObjectGenerationProfile>(true);

            if (profile != null)
            {
                return profile.GetRestingPoseWeights();
            }
        }

        return fallbackWeights;
    }

    public void ResetSamplingState()
    {
        classCacheBuilt = false;
        prefabClasses.Clear();
        semanticClasses.Clear();
        shuffledClassBag.Clear();
    }

    private bool BuildClassCache()
    {
        prefabClasses.Clear();
        semanticClasses.Clear();
        shuffledClassBag.Clear();

        Dictionary<string, List<GameObject>> groupedPrefabs =
            new Dictionary<string, List<GameObject>>();

        foreach (GameObject prefab in prefabs)
        {
            if (prefab == null)
            {
                Debug.LogError("Perception object library contains a null prefab.");
                return false;
            }

            if (!TryGetSemanticClass(prefab, out string semanticClass))
            {
                Debug.LogError(
                    $"Target prefab '{prefab.name}' must contain exactly one " +
                    "non-empty semantic class across its Perception Labeling components."
                );
                return false;
            }

            if (!groupedPrefabs.TryGetValue(
                semanticClass,
                out List<GameObject> classPrefabs
            ))
            {
                classPrefabs = new List<GameObject>();
                groupedPrefabs.Add(semanticClass, classPrefabs);
            }

            classPrefabs.Add(prefab);
        }

        List<string> sortedClassNames = new List<string>(groupedPrefabs.Keys);
        sortedClassNames.Sort(System.StringComparer.Ordinal);

        foreach (string semanticClass in sortedClassNames)
        {
            semanticClasses.Add(semanticClass);
            prefabClasses.Add(groupedPrefabs[semanticClass]);
        }

        if (prefabClasses.Count == 0)
        {
            Debug.LogError("Perception object library has no semantic classes.");
            return false;
        }

        classCacheBuilt = true;

        Debug.Log(
            $"Perception object library balanced across {semanticClasses.Count} " +
            $"semantic classes: {string.Join(", ", semanticClasses)}"
        );

        return true;
    }

    private bool TryGetSemanticClass(
        GameObject prefab,
        out string semanticClass
    )
    {
        HashSet<string> labels = new HashSet<string>();
        Labeling[] labelings = prefab.GetComponentsInChildren<Labeling>(true);

        foreach (Labeling labeling in labelings)
        {
            foreach (string label in labeling.labels)
            {
                if (!string.IsNullOrWhiteSpace(label))
                {
                    labels.Add(label.Trim());
                }
            }
        }

        if (labels.Count != 1)
        {
            semanticClass = null;
            return false;
        }

        foreach (string label in labels)
        {
            semanticClass = label;
            return true;
        }

        semanticClass = null;
        return false;
    }

    private void RefillClassBag()
    {
        shuffledClassBag.Clear();

        for (int i = 0; i < prefabClasses.Count; i++)
        {
            shuffledClassBag.Add(i);
        }

        for (int i = shuffledClassBag.Count - 1; i > 0; i--)
        {
            int swapIndex = Random.Range(0, i + 1);
            int temporary = shuffledClassBag[i];
            shuffledClassBag[i] = shuffledClassBag[swapIndex];
            shuffledClassBag[swapIndex] = temporary;
        }
    }
}
