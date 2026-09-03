using UnityEngine;

public class MaterialRandomizer : MonoBehaviour
{
    [Header("Renderers")]
    public Renderer tableRenderer;
    public Renderer floorRenderer;
    public Renderer[] wallRenderers;

    [Header("Material Options")]
    public Material[] tableMaterialOptions;
    public Material[] floorMaterialOptions;
    public Material[] wallMaterialOptions;

    [Header("Wall Behaviour")]
    public bool useSameWallMaterialForAllWalls = true;

    [Header("Texture Tiling")]
    public bool randomizeTextureTiling = true;
    public Vector2 floorTilingRange = new Vector2(1.0f, 3.0f);
    public Vector2 wallTilingRange = new Vector2(1.0f, 2.5f);
    public Vector2 tableTilingRange = new Vector2(1.0f, 2.0f);

    [Header("Table Color Fallback")]
    public Color[] tableColors =
    {
        new Color(0.72f, 0.68f, 0.61f),
        new Color(0.55f, 0.50f, 0.44f),
        new Color(0.80f, 0.78f, 0.72f),
        new Color(0.35f, 0.32f, 0.28f)
    };

    [Header("Floor Color Fallback")]
    public Color[] floorColors =
    {
        new Color(0.45f, 0.45f, 0.45f),
        new Color(0.60f, 0.60f, 0.58f),
        new Color(0.32f, 0.34f, 0.36f)
    };

    [Header("Wall Color Fallback")]
    public Color[] wallColors =
    {
        new Color(0.85f, 0.85f, 0.82f),
        new Color(0.75f, 0.78f, 0.80f),
        new Color(0.90f, 0.88f, 0.84f),
        new Color(0.70f, 0.72f, 0.74f)
    };

    public void RandomizeMaterials()
    {
        ApplyRandomMaterialOrColor(
            tableRenderer,
            tableMaterialOptions,
            tableColors,
            tableTilingRange
        );

        ApplyRandomMaterialOrColor(
            floorRenderer,
            floorMaterialOptions,
            floorColors,
            floorTilingRange
        );

        if (wallRenderers == null || wallRenderers.Length == 0)
        {
            return;
        }

        if (useSameWallMaterialForAllWalls)
        {
            Material selectedWallMaterial = GetRandomMaterial(wallMaterialOptions);
            Color selectedWallColor = GetRandomColor(wallColors);
            float selectedWallTiling = GetRandomTiling(wallTilingRange);

            foreach (Renderer wallRenderer in wallRenderers)
            {
                ApplyMaterialOrColor(
                    wallRenderer,
                    selectedWallMaterial,
                    selectedWallColor,
                    selectedWallTiling
                );
            }
        }
        else
        {
            foreach (Renderer wallRenderer in wallRenderers)
            {
                ApplyRandomMaterialOrColor(
                    wallRenderer,
                    wallMaterialOptions,
                    wallColors,
                    wallTilingRange
                );
            }
        }
    }

    private void ApplyRandomMaterialOrColor(
        Renderer targetRenderer,
        Material[] materialOptions,
        Color[] colorFallbacks,
        Vector2 tilingRange
    )
    {
        ApplyMaterialOrColor(
            targetRenderer,
            GetRandomMaterial(materialOptions),
            GetRandomColor(colorFallbacks),
            GetRandomTiling(tilingRange)
        );
    }

    private void ApplyMaterialOrColor(
        Renderer targetRenderer,
        Material selectedMaterial,
        Color selectedColor,
        float tiling
    )
    {
        if (targetRenderer == null)
        {
            return;
        }

        Material materialInstance;

        if (selectedMaterial != null)
        {
            targetRenderer.material = selectedMaterial;
            materialInstance = targetRenderer.material;
        }
        else
        {
            materialInstance = targetRenderer.material;
            ApplyColor(materialInstance, selectedColor);
        }

        if (randomizeTextureTiling)
        {
            ApplyTextureTiling(materialInstance, tiling);
        }
    }

    private Material GetRandomMaterial(Material[] materialOptions)
    {
        if (materialOptions == null || materialOptions.Length == 0)
        {
            return null;
        }

        return materialOptions[Random.Range(0, materialOptions.Length)];
    }

    private Color GetRandomColor(Color[] colors)
    {
        if (colors == null || colors.Length == 0)
        {
            return Color.white;
        }

        return colors[Random.Range(0, colors.Length)];
    }

    private float GetRandomTiling(Vector2 tilingRange)
    {
        float min = Mathf.Max(0.01f, Mathf.Min(tilingRange.x, tilingRange.y));
        float max = Mathf.Max(min, Mathf.Max(tilingRange.x, tilingRange.y));
        return Random.Range(min, max);
    }

    private void ApplyColor(Material material, Color color)
    {
        if (material == null)
        {
            return;
        }

        if (material.HasProperty("_BaseColor"))
        {
            material.SetColor("_BaseColor", color);
        }
        else if (material.HasProperty("_Color"))
        {
            material.SetColor("_Color", color);
        }
        else
        {
            material.color = color;
        }
    }

    private void ApplyTextureTiling(Material material, float tiling)
    {
        if (material == null)
        {
            return;
        }

        Vector2 scale = new Vector2(tiling, tiling);

        if (material.HasProperty("_BaseColorMap"))
        {
            material.SetTextureScale("_BaseColorMap", scale);
        }

        if (material.HasProperty("_MainTex"))
        {
            material.SetTextureScale("_MainTex", scale);
        }
    }
}
