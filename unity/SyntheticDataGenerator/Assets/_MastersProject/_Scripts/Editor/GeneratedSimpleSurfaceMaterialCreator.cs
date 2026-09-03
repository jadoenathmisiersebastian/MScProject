using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class GeneratedSimpleSurfaceMaterialCreator
{
    private const int TextureSize = 512;

    private sealed class SurfaceSpec
    {
        public readonly string Surface;
        public readonly string Name;
        public readonly Color BaseColor;
        public readonly Color GrainColor;
        public readonly float Smoothness;
        public readonly int Seed;

        public SurfaceSpec(
            string surface,
            string name,
            Color baseColor,
            Color grainColor,
            float smoothness,
            int seed
        )
        {
            Surface = surface;
            Name = name;
            BaseColor = baseColor;
            GrainColor = grainColor;
            Smoothness = smoothness;
            Seed = seed;
        }
    }

    private static readonly SurfaceSpec[] Specs =
    {
        new SurfaceSpec(
            "Floor", "Floor_SimpleWhite",
            Rgb(224, 224, 218), Rgb(190, 193, 190), 0.28f, 501
        ),
        new SurfaceSpec(
            "Floor", "Floor_SimpleBeige",
            Rgb(190, 174, 145), Rgb(151, 137, 113), 0.24f, 502
        ),
        new SurfaceSpec(
            "Floor", "Floor_SimpleNearBlack",
            Rgb(34, 35, 35), Rgb(61, 62, 60), 0.22f, 503
        ),
        new SurfaceSpec(
            "Table", "Table_SimpleWhite",
            Rgb(232, 231, 223), Rgb(198, 200, 197), 0.44f, 601
        ),
        new SurfaceSpec(
            "Table", "Table_SimpleBeige",
            Rgb(201, 184, 151), Rgb(164, 146, 118), 0.38f, 602
        ),
        new SurfaceSpec(
            "Table", "Table_SimpleNearBlack",
            Rgb(30, 31, 31), Rgb(57, 58, 56), 0.40f, 603
        )
    };

    [MenuItem("MastersProject/Materials/Create and Add Simple Floor/Table Pack")]
    public static void CreateAndAddSimpleSurfacePack()
    {
        Shader shader = Shader.Find("HDRP/Lit");
        if (shader == null)
        {
            Debug.LogError("HDRP/Lit was not found. Confirm HDRP is active.");
            return;
        }

        List<Material> floorMaterials = new List<Material>();
        List<Material> tableMaterials = new List<Material>();

        foreach (SurfaceSpec spec in Specs)
        {
            string folder =
                $"Assets/_MastersProject/Materials/Textures/{spec.Surface}/SimpleGenerated";
            EnsureFolder(folder);

            string texturePath = $"{folder}/T_{spec.Name}.png";
            CreateOrUpdateTexture(spec, texturePath);
            Texture2D texture = AssetDatabase.LoadAssetAtPath<Texture2D>(texturePath);
            if (texture == null)
            {
                Debug.LogError($"Could not import generated texture {texturePath}.");
                continue;
            }

            string materialPath = $"{folder}/M_{spec.Name}.mat";
            Material material = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
            if (material == null)
            {
                material = new Material(shader);
                AssetDatabase.CreateAsset(material, materialPath);
            }
            else
            {
                material.shader = shader;
            }

            ConfigureMaterial(material, texture, spec.Smoothness);
            EditorUtility.SetDirty(material);

            if (spec.Surface == "Floor")
            {
                floorMaterials.Add(material);
            }
            else
            {
                tableMaterials.Add(material);
            }
        }

        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();

        int randomizersUpdated = AppendToLoadedRandomizers(
            floorMaterials.ToArray(),
            tableMaterials.ToArray()
        );

        Debug.Log(
            $"Created six simple floor/table materials and updated " +
            $"{randomizersUpdated} loaded MaterialRandomizer component(s)."
        );
    }

    private static void CreateOrUpdateTexture(SurfaceSpec spec, string path)
    {
        Texture2D texture = new Texture2D(
            TextureSize,
            TextureSize,
            TextureFormat.RGBA32,
            true,
            false
        );
        texture.name = $"T_{spec.Name}";
        texture.wrapMode = TextureWrapMode.Repeat;
        texture.filterMode = FilterMode.Trilinear;

        Color32[] pixels = new Color32[TextureSize * TextureSize];
        System.Random random = new System.Random(spec.Seed);

        for (int y = 0; y < TextureSize; y++)
        {
            float v = y / (float)TextureSize;
            for (int x = 0; x < TextureSize; x++)
            {
                float u = x / (float)TextureSize;
                float fineNoise = ((float)random.NextDouble() - 0.5f) * 0.024f;
                float broadNoise =
                    Mathf.Sin((u * 5f + v * 3f) * Mathf.PI + spec.Seed) * 0.012f +
                    Mathf.Sin((u * 11f - v * 7f) * Mathf.PI) * 0.006f;
                float grain = Mathf.Clamp01(0.08f + fineNoise + broadNoise);
                Color color = Color.Lerp(spec.BaseColor, spec.GrainColor, grain);
                color.a = 1f;
                pixels[y * TextureSize + x] = color;
            }
        }

        texture.SetPixels32(pixels);
        texture.Apply(true, false);
        File.WriteAllBytes(path, texture.EncodeToPNG());
        UnityEngine.Object.DestroyImmediate(texture);
        AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceUpdate);

        TextureImporter importer = AssetImporter.GetAtPath(path) as TextureImporter;
        if (importer != null)
        {
            importer.textureType = TextureImporterType.Default;
            importer.sRGBTexture = true;
            importer.wrapMode = TextureWrapMode.Repeat;
            importer.filterMode = FilterMode.Trilinear;
            importer.mipmapEnabled = true;
            importer.textureCompression = TextureImporterCompression.CompressedHQ;
            importer.SaveAndReimport();
        }
    }

    private static void ConfigureMaterial(
        Material material,
        Texture2D texture,
        float smoothness
    )
    {
        SetTexture(material, "_BaseColorMap", texture);
        SetTexture(material, "_MainTex", texture);
        SetColor(material, "_BaseColor", Color.white);
        SetColor(material, "_Color", Color.white);
        SetFloat(material, "_Metallic", 0f);
        SetFloat(material, "_Smoothness", smoothness);
        SetFloat(material, "_CoatMask", 0.02f);
        SetFloat(material, "_SurfaceType", 0f);
        SetFloat(material, "_ZWrite", 1f);
        material.SetOverrideTag("RenderType", string.Empty);
        material.DisableKeyword("_SURFACE_TYPE_TRANSPARENT");
        material.renderQueue = -1;
    }

    private static int AppendToLoadedRandomizers(
        Material[] floorMaterials,
        Material[] tableMaterials
    )
    {
        MaterialRandomizer[] randomizers =
            Resources.FindObjectsOfTypeAll<MaterialRandomizer>();
        int updated = 0;

        foreach (MaterialRandomizer randomizer in randomizers)
        {
            if (randomizer == null ||
                EditorUtility.IsPersistent(randomizer) ||
                !randomizer.gameObject.scene.IsValid())
            {
                continue;
            }

            Undo.RecordObject(randomizer, "Add simple surface materials");
            randomizer.floorMaterialOptions = AppendUnique(
                randomizer.floorMaterialOptions,
                floorMaterials
            );
            randomizer.tableMaterialOptions = AppendUnique(
                randomizer.tableMaterialOptions,
                tableMaterials
            );
            EditorUtility.SetDirty(randomizer);
            EditorSceneManager.MarkSceneDirty(randomizer.gameObject.scene);
            updated++;
        }

        return updated;
    }

    private static Material[] AppendUnique(Material[] existing, Material[] additions)
    {
        List<Material> result = existing == null
            ? new List<Material>()
            : new List<Material>(existing);

        foreach (Material material in additions)
        {
            if (material != null && !result.Contains(material))
            {
                result.Add(material);
            }
        }
        return result.ToArray();
    }

    private static void SetTexture(Material material, string property, Texture value)
    {
        if (material.HasProperty(property))
        {
            material.SetTexture(property, value);
            material.SetTextureScale(property, Vector2.one);
        }
    }

    private static void SetColor(Material material, string property, Color value)
    {
        if (material.HasProperty(property))
        {
            material.SetColor(property, value);
        }
    }

    private static void SetFloat(Material material, string property, float value)
    {
        if (material.HasProperty(property))
        {
            material.SetFloat(property, value);
        }
    }

    private static Color Rgb(byte red, byte green, byte blue)
    {
        return new Color32(red, green, blue, 255);
    }

    private static void EnsureFolder(string path)
    {
        if (AssetDatabase.IsValidFolder(path))
        {
            return;
        }
        string parent = Path.GetDirectoryName(path).Replace("\\", "/");
        EnsureFolder(parent);
        AssetDatabase.CreateFolder(parent, Path.GetFileName(path));
    }
}
