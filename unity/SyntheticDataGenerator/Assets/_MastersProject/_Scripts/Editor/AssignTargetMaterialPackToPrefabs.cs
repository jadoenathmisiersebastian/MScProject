using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.Perception.GroundTruth.LabelManagement;

public static class AssignTargetMaterialPackToPrefabs
{
    private const string PrefabFolder =
        "Assets/_MastersProject/Prefabs/Items";

    private const string MaterialRoot =
        "Assets/_MastersProject/Materials/Targets/ClassSpecificGenerated";

    private static readonly string[] ExcludedRendererNameParts =
    {
        "label",
        "liquid",
        "wrapper",
        "lid",
        "cap",
        "cork",
        "contents"
    };

    [MenuItem("MastersProject/Materials/Assign Class-Specific Pack to Item Prefabs")]
    public static void AssignMaterialPack()
    {
        Dictionary<string, Material[]> materialsByClass =
            BuildMaterialsByClass();

        if (materialsByClass.Count == 0)
        {
            Debug.LogError(
                "No class-specific target materials were found. Run " +
                "MastersProject > Materials > Create Class-Specific Target " +
                "Pack first."
            );
            return;
        }

        string[] prefabGuids = AssetDatabase.FindAssets(
            "t:Prefab",
            new[] { PrefabFolder }
        );

        int updated = 0;
        int skipped = 0;

        foreach (string prefabGuid in prefabGuids)
        {
            string prefabPath = AssetDatabase.GUIDToAssetPath(prefabGuid);
            GameObject prefabRoot = PrefabUtility.LoadPrefabContents(prefabPath);

            try
            {
                string semanticClass = GetSemanticClass(prefabRoot);

                if (string.IsNullOrEmpty(semanticClass) ||
                    !materialsByClass.TryGetValue(
                        semanticClass,
                        out Material[] materialOptions
                    ))
                {
                    skipped++;
                    continue;
                }

                Renderer targetRenderer = FindPrimaryBodyRenderer(prefabRoot);

                if (targetRenderer == null ||
                    targetRenderer.sharedMaterials == null ||
                    targetRenderer.sharedMaterials.Length == 0)
                {
                    Debug.LogWarning(
                        $"Skipped '{prefabRoot.name}': no suitable body renderer " +
                        "with a material slot was found."
                    );
                    skipped++;
                    continue;
                }

                TargetAppearanceRandomizer randomizer =
                    prefabRoot.GetComponent<TargetAppearanceRandomizer>();

                if (randomizer == null)
                {
                    randomizer =
                        prefabRoot.AddComponent<TargetAppearanceRandomizer>();
                }

                Material originalMaterial = targetRenderer.sharedMaterials[0];

                randomizer.materialSlots = new[]
                {
                    new TargetAppearanceRandomizer.MaterialSlot
                    {
                        targetRenderer = targetRenderer,
                        materialIndex = 0,
                        options = BuildWeightedOptions(
                            originalMaterial,
                            materialOptions
                        )
                    }
                };

                randomizer.randomizeOnAwake = true;
                EditorUtility.SetDirty(randomizer);
                PrefabUtility.SaveAsPrefabAsset(prefabRoot, prefabPath);
                updated++;

                Debug.Log(
                    $"Configured '{prefabRoot.name}' ({semanticClass}) using " +
                    $"renderer '{targetRenderer.name}', material index 0, and " +
                    $"{randomizer.materialSlots[0].options.Length} options."
                );
            }
            finally
            {
                PrefabUtility.UnloadPrefabContents(prefabRoot);
            }
        }

        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();

        Debug.Log(
            $"Target material assignment complete: {updated} prefabs updated, " +
            $"{skipped} prefabs skipped."
        );
    }

    private static Dictionary<string, Material[]> BuildMaterialsByClass()
    {
        Dictionary<string, Material[]> output =
            new Dictionary<string, Material[]>();

        Material[] drinkCartonMaterials = LoadMaterials("DrinkCartons");
        Material[] bottleMaterials = LoadMaterials("Bottles");
        Material[] glassMaterials = LoadMaterials("Glasses");
        Material[] foodBoxMaterials = LoadMaterials("FoodBoxes");

        if (drinkCartonMaterials.Length > 0)
        {
            output["drink_carton"] = drinkCartonMaterials;
        }

        if (bottleMaterials.Length > 0)
        {
            output["bottle"] = bottleMaterials;
        }

        if (glassMaterials.Length > 0)
        {
            output["glass"] = glassMaterials;
        }

        if (foodBoxMaterials.Length > 0)
        {
            output["food_box"] = foodBoxMaterials;
        }

        return output;
    }

    private static Material[] LoadMaterials(string category)
    {
        string folder = $"{MaterialRoot}/{category}";
        string[] guids = AssetDatabase.FindAssets(
            "t:Material",
            new[] { folder }
        );

        List<Material> materials = new List<Material>();

        foreach (string guid in guids)
        {
            Material material = AssetDatabase.LoadAssetAtPath<Material>(
                AssetDatabase.GUIDToAssetPath(guid)
            );

            if (material != null)
            {
                materials.Add(material);
            }
        }

        materials.Sort(
            (left, right) => string.Compare(
                left.name,
                right.name,
                StringComparison.Ordinal
            )
        );

        return materials.ToArray();
    }

    private static string GetSemanticClass(GameObject prefabRoot)
    {
        Labeling[] labelings =
            prefabRoot.GetComponentsInChildren<Labeling>(true);

        HashSet<string> labels = new HashSet<string>();

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
            return null;
        }

        foreach (string label in labels)
        {
            return label;
        }

        return null;
    }

    private static Renderer FindPrimaryBodyRenderer(GameObject prefabRoot)
    {
        Renderer[] renderers =
            prefabRoot.GetComponentsInChildren<Renderer>(true);

        Renderer bestRenderer = null;
        float bestVolume = float.NegativeInfinity;

        foreach (Renderer renderer in renderers)
        {
            if (!(renderer is MeshRenderer) &&
                !(renderer is SkinnedMeshRenderer))
            {
                continue;
            }

            if (IsExcludedRenderer(renderer.name))
            {
                continue;
            }

            Material[] materials = renderer.sharedMaterials;

            if (materials == null || materials.Length == 0)
            {
                continue;
            }

            Vector3 size = renderer.bounds.size;
            float volume = Mathf.Abs(size.x * size.y * size.z);

            if (bestRenderer == null || volume > bestVolume)
            {
                bestRenderer = renderer;
                bestVolume = volume;
            }
        }

        return bestRenderer;
    }

    private static bool IsExcludedRenderer(string rendererName)
    {
        string normalizedName = rendererName.ToLowerInvariant();

        foreach (string excludedPart in ExcludedRendererNameParts)
        {
            if (normalizedName.Contains(excludedPart))
            {
                return true;
            }
        }

        return false;
    }

    private static TargetAppearanceRandomizer.WeightedMaterialOption[]
        BuildWeightedOptions(
            Material originalMaterial,
            Material[] generatedMaterials
        )
    {
        List<TargetAppearanceRandomizer.WeightedMaterialOption> options =
            new List<TargetAppearanceRandomizer.WeightedMaterialOption>();

        if (originalMaterial != null)
        {
            options.Add(
                new TargetAppearanceRandomizer.WeightedMaterialOption
                {
                    material = originalMaterial,
                    weight = 2f
                }
            );
        }

        foreach (Material material in generatedMaterials)
        {
            if (material == null || material == originalMaterial)
            {
                continue;
            }

            options.Add(
                new TargetAppearanceRandomizer.WeightedMaterialOption
                {
                    material = material,
                    weight = 1f
                }
            );
        }

        return options.ToArray();
    }
}
