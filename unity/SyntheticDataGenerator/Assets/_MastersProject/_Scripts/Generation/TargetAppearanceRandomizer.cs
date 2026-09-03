using System;
using UnityEngine;

[DisallowMultipleComponent]
public class TargetAppearanceRandomizer : MonoBehaviour
{
    [Serializable]
    public class WeightedMaterialOption
    {
        public Material material;

        [Min(0f)]
        public float weight = 1f;
    }

    [Serializable]
    public class MaterialSlot
    {
        public Renderer targetRenderer;

        [Min(0)]
        public int materialIndex;

        public WeightedMaterialOption[] options;
    }

    [Header("Material Variants")]
    public MaterialSlot[] materialSlots;

    [Header("Behaviour")]
    public bool randomizeOnAwake = true;

    private void Awake()
    {
        if (randomizeOnAwake)
        {
            RandomizeAppearance();
        }
    }

    public void RandomizeAppearance()
    {
        if (materialSlots == null)
        {
            return;
        }

        foreach (MaterialSlot slot in materialSlots)
        {
            ApplyRandomMaterial(slot);
        }
    }

    private void ApplyRandomMaterial(MaterialSlot slot)
    {
        if (slot == null || slot.targetRenderer == null)
        {
            return;
        }

        Material selectedMaterial = SelectWeightedMaterial(slot.options);

        if (selectedMaterial == null)
        {
            return;
        }

        Material[] materials = slot.targetRenderer.sharedMaterials;

        if (slot.materialIndex < 0 || slot.materialIndex >= materials.Length)
        {
            Debug.LogWarning(
                $"Invalid material index {slot.materialIndex} on " +
                $"{slot.targetRenderer.name}.",
                this
            );
            return;
        }

        materials[slot.materialIndex] = selectedMaterial;
        slot.targetRenderer.sharedMaterials = materials;
    }

    private Material SelectWeightedMaterial(
        WeightedMaterialOption[] options
    )
    {
        if (options == null || options.Length == 0)
        {
            return null;
        }

        float totalWeight = 0f;

        foreach (WeightedMaterialOption option in options)
        {
            if (option != null && option.material != null)
            {
                totalWeight += Mathf.Max(0f, option.weight);
            }
        }

        if (totalWeight <= 0f)
        {
            return null;
        }

        float selection = UnityEngine.Random.value * totalWeight;

        foreach (WeightedMaterialOption option in options)
        {
            if (option == null || option.material == null)
            {
                continue;
            }

            selection -= Mathf.Max(0f, option.weight);

            if (selection <= 0f)
            {
                return option.material;
            }
        }

        return null;
    }
}