using System;
using System.IO;
using UnityEditor;
using UnityEngine;

public static class GeneratedTargetMaterialPackCreator
{
    private const int TextureSize = 512;

    private const string MaterialRoot =
        "Assets/_MastersProject/Materials/Targets/ClassSpecificGenerated";

    private const string TextureRoot =
        "Assets/_MastersProject/Materials/Textures/Targets/ClassSpecificGenerated";

    private enum PatternType
    {
        Carton,
        FoodBox,
        Kraft,
        BottleGlass,
        BottlePlastic,
        DrinkingGlass
    }

    private sealed class MaterialSpec
    {
        public readonly string Category;
        public readonly string Name;
        public readonly Color Primary;
        public readonly Color Secondary;
        public readonly Color Accent;
        public readonly float Smoothness;
        public readonly float CoatMask;
        public readonly PatternType Pattern;
        public readonly int Seed;
        public readonly bool Transparent;
        public readonly float Alpha;

        public MaterialSpec(
            string category,
            string name,
            Color primary,
            Color secondary,
            Color accent,
            float smoothness,
            float coatMask,
            PatternType pattern,
            int seed,
            bool transparent = false,
            float alpha = 1f
        )
        {
            Category = category;
            Name = name;
            Primary = primary;
            Secondary = secondary;
            Accent = accent;
            Smoothness = smoothness;
            CoatMask = coatMask;
            Pattern = pattern;
            Seed = seed;
            Transparent = transparent;
            Alpha = alpha;
        }
    }

    private static readonly MaterialSpec[] MaterialSpecs =
    {
        // Opaque coated card and plastic-coated drink cartons.
        Spec("DrinkCartons", "DrinkCarton_DairyBlue", 232, 231, 220, 56, 105, 154, 31, 53, 79, 0.38f, 0.08f, PatternType.Carton, 101),
        Spec("DrinkCartons", "DrinkCarton_FruitOrange", 239, 224, 185, 215, 117, 42, 69, 116, 64, 0.34f, 0.06f, PatternType.Carton, 102),
        Spec("DrinkCartons", "DrinkCarton_NaturalGreen", 220, 219, 191, 76, 126, 79, 83, 61, 42, 0.28f, 0.04f, PatternType.Carton, 103),
        Spec("DrinkCartons", "DrinkCarton_BerryRed", 233, 218, 206, 166, 48, 58, 87, 39, 75, 0.39f, 0.08f, PatternType.Carton, 104),
        Spec("DrinkCartons", "DrinkCarton_CocoaCream", 224, 205, 175, 105, 61, 39, 177, 54, 43, 0.33f, 0.06f, PatternType.Carton, 105),
        Spec("DrinkCartons", "DrinkCarton_KraftBlue", 170, 132, 84, 44, 91, 132, 226, 216, 189, 0.22f, 0.02f, PatternType.Kraft, 106),

        // Hard glass remains opaque for reliable labels and depth, but uses a
        // high-smoothness coated finish. PET options are slightly rougher.
        Spec("Bottles", "Bottle_AmberHardGlass", 76, 35, 16, 137, 72, 28, 224, 201, 151, 0.94f, 0.82f, PatternType.BottleGlass, 201, true, 0.80f),
        Spec("Bottles", "Bottle_GreenHardGlass", 18, 67, 43, 48, 105, 65, 218, 211, 169, 0.94f, 0.82f, PatternType.BottleGlass, 202, true, 0.78f),
        Spec("Bottles", "Bottle_BlueHardGlass", 24, 65, 102, 58, 113, 143, 224, 224, 204, 0.93f, 0.78f, PatternType.BottleGlass, 203, true, 0.74f),
        Spec("Bottles", "Bottle_ClearFrostedGlass", 187, 201, 199, 226, 231, 222, 95, 126, 126, 0.82f, 0.62f, PatternType.BottleGlass, 204, true, 0.84f),
        Spec("Bottles", "Bottle_WhitePET", 221, 223, 216, 175, 187, 186, 42, 104, 149, 0.62f, 0.18f, PatternType.BottlePlastic, 205),
        Spec("Bottles", "Bottle_ClearPET", 184, 205, 211, 224, 232, 229, 38, 111, 147, 0.72f, 0.24f, PatternType.BottlePlastic, 206),
        Spec("Bottles", "Bottle_AquaPET", 41, 132, 137, 96, 181, 171, 229, 232, 213, 0.68f, 0.22f, PatternType.BottlePlastic, 207),
        Spec("Bottles", "Bottle_RedPET", 156, 39, 43, 203, 82, 67, 237, 220, 174, 0.66f, 0.20f, PatternType.BottlePlastic, 208),

        // A restrained tint range for transparent drinking glasses.
        Spec("Glasses", "Glass_Clear", 221, 230, 228, 244, 246, 240, 186, 208, 207, 0.96f, 0.82f, PatternType.DrinkingGlass, 301, true, 0.22f),
        Spec("Glasses", "Glass_Smoke", 84, 91, 94, 146, 151, 149, 213, 217, 211, 0.95f, 0.78f, PatternType.DrinkingGlass, 302, true, 0.30f),
        Spec("Glasses", "Glass_PaleBlue", 116, 169, 188, 192, 217, 218, 231, 237, 228, 0.95f, 0.80f, PatternType.DrinkingGlass, 303, true, 0.27f),
        Spec("Glasses", "Glass_PaleGreen", 112, 163, 132, 190, 212, 187, 232, 232, 213, 0.95f, 0.80f, PatternType.DrinkingGlass, 304, true, 0.27f),

        // Opaque card food packaging with varied print layouts.
        Spec("FoodBoxes", "FoodBox_CerealBlue", 47, 91, 143, 226, 207, 112, 238, 231, 211, 0.38f, 0.08f, PatternType.FoodBox, 401),
        Spec("FoodBoxes", "FoodBox_CerealYellow", 222, 171, 50, 86, 58, 42, 190, 47, 41, 0.36f, 0.07f, PatternType.FoodBox, 402),
        Spec("FoodBoxes", "FoodBox_PastaRed", 174, 40, 39, 233, 218, 179, 51, 80, 113, 0.40f, 0.09f, PatternType.FoodBox, 403),
        Spec("FoodBoxes", "FoodBox_SnackGreen", 61, 116, 72, 229, 212, 143, 86, 54, 38, 0.35f, 0.06f, PatternType.FoodBox, 404),
        Spec("FoodBoxes", "FoodBox_CoatedWhite", 229, 226, 214, 68, 106, 141, 177, 49, 47, 0.42f, 0.10f, PatternType.FoodBox, 405),
        Spec("FoodBoxes", "FoodBox_RecycledKraft", 163, 124, 77, 103, 76, 47, 220, 210, 177, 0.18f, 0.01f, PatternType.Kraft, 406)
    };

    [MenuItem("MastersProject/Materials/Create Class-Specific Target Pack")]
    public static void CreateTargetMaterialPack()
    {
        Shader shader = Shader.Find("HDRP/Lit");
        if (shader == null)
        {
            Debug.LogError("HDRP/Lit was not found. Confirm HDRP is active.");
            return;
        }

        EnsureFolder(MaterialRoot);
        EnsureFolder(TextureRoot);
        int createdOrUpdated = 0;

        foreach (MaterialSpec spec in MaterialSpecs)
        {
            string materialFolder = $"{MaterialRoot}/{spec.Category}";
            string textureFolder = $"{TextureRoot}/{spec.Category}";
            EnsureFolder(materialFolder);
            EnsureFolder(textureFolder);

            string texturePath = $"{textureFolder}/T_{spec.Name}.png";
            CreateOrUpdateTexture(spec, texturePath);
            Texture2D texture = AssetDatabase.LoadAssetAtPath<Texture2D>(texturePath);
            if (texture == null)
            {
                Debug.LogError($"Could not import generated texture {texturePath}.");
                continue;
            }

            string materialPath = $"{materialFolder}/M_{spec.Name}.mat";
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

            ConfigureMaterial(material, texture, spec);
            EditorUtility.SetDirty(material);
            createdOrUpdated++;
        }

        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
        Debug.Log($"Created/updated {createdOrUpdated} materials under {MaterialRoot}.");
    }

    private static MaterialSpec Spec(
        string category,
        string name,
        byte pr, byte pg, byte pb,
        byte sr, byte sg, byte sb,
        byte ar, byte ag, byte ab,
        float smoothness,
        float coatMask,
        PatternType pattern,
        int seed,
        bool transparent = false,
        float alpha = 1f
    )
    {
        return new MaterialSpec(
            category,
            name,
            Rgb(pr, pg, pb),
            Rgb(sr, sg, sb),
            Rgb(ar, ag, ab),
            smoothness,
            coatMask,
            pattern,
            seed,
            transparent,
            alpha
        );
    }

    private static void CreateOrUpdateTexture(MaterialSpec spec, string path)
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
            for (int x = 0; x < TextureSize; x++)
            {
                pixels[y * TextureSize + x] = BuildPixel(spec, x, y, random);
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

    private static Color32 BuildPixel(
        MaterialSpec spec,
        int x,
        int y,
        System.Random random
    )
    {
        float u = x / (float)TextureSize;
        float v = y / (float)TextureSize;
        float noise = ((float)random.NextDouble() - 0.5f) * 0.032f;
        float wave =
            Mathf.Sin((u * 13f + v * 7f) * Mathf.PI + spec.Seed) * 0.5f +
            Mathf.Sin((u * 5f - v * 17f) * Mathf.PI) * 0.5f;
        Color color;

        switch (spec.Pattern)
        {
            case PatternType.Carton:
                color = BuildCarton(spec, u, v, noise);
                break;
            case PatternType.FoodBox:
                color = BuildFoodBox(spec, u, v, noise);
                break;
            case PatternType.Kraft:
                color = BuildKraft(spec, u, v, wave, noise);
                break;
            case PatternType.BottleGlass:
                color = BuildBottle(spec, u, v, wave, true);
                break;
            case PatternType.BottlePlastic:
                color = BuildBottle(spec, u, v, wave, false);
                break;
            case PatternType.DrinkingGlass:
                color = BuildGlass(spec, u, v, wave, noise);
                break;
            default:
                color = spec.Primary;
                break;
        }

        color.r = Mathf.Clamp01(color.r);
        color.g = Mathf.Clamp01(color.g);
        color.b = Mathf.Clamp01(color.b);
        color.a = 1f;
        return color;
    }

    private static Color BuildCarton(MaterialSpec spec, float u, float v, float noise)
    {
        Color color = spec.Primary * (1f + noise * 0.7f);
        if (v > 0.72f && v < 0.88f)
        {
            color = spec.Secondary;
        }
        else if (v < 0.13f)
        {
            color = Color.Lerp(spec.Secondary, spec.Accent, 0.42f);
        }

        float panel = Mathf.Repeat(u * 2f, 1f);
        if (panel < 0.035f || panel > 0.965f)
        {
            color = Color.Lerp(color, spec.Accent, 0.42f);
        }

        return IsPseudoPrint(u, v, spec.Seed, 7f, 15f)
            ? Color.Lerp(color, spec.Accent, 0.82f)
            : color;
    }

    private static Color BuildFoodBox(MaterialSpec spec, float u, float v, float noise)
    {
        Color color = spec.Primary * (1f + noise * 0.6f);
        if (u > 0.17f && u < 0.83f && v > 0.18f && v < 0.80f)
        {
            color = Color.Lerp(spec.Primary, spec.Secondary, 0.72f);
        }
        if ((v > 0.79f && v < 0.89f) || (v > 0.08f && v < 0.15f))
        {
            color = spec.Accent;
        }

        float logo = Vector2.Distance(new Vector2(u, v), new Vector2(0.5f, 0.60f));
        if (logo < 0.11f)
        {
            color = Color.Lerp(spec.Secondary, spec.Accent, logo * 5f);
        }

        return IsPseudoPrint(u, v, spec.Seed, 8f, 17f)
            ? Color.Lerp(color, spec.Accent, 0.86f)
            : color;
    }

    private static Color BuildKraft(
        MaterialSpec spec,
        float u,
        float v,
        float wave,
        float noise
    )
    {
        float fiber = Mathf.Pow(Mathf.Abs(Mathf.Sin(v * Mathf.PI * 73f)), 22f) * 0.13f;
        Color color = Color.Lerp(spec.Primary, spec.Secondary, fiber + 0.08f);
        color *= 1f + wave * 0.035f + noise;
        return (v > 0.69f && v < 0.82f) || IsPseudoPrint(u, v, spec.Seed, 6f, 13f)
            ? Color.Lerp(color, spec.Accent, 0.72f)
            : color;
    }

    private static Color BuildBottle(
        MaterialSpec spec,
        float u,
        float v,
        float wave,
        bool hardGlass
    )
    {
        float highlight = Mathf.Pow(
            Mathf.Max(0f, Mathf.Sin(u * Mathf.PI * 2f)),
            hardGlass ? 8f : 4f
        );
        Color color = Color.Lerp(
            spec.Primary,
            spec.Secondary,
            0.10f + highlight * (hardGlass ? 0.22f : 0.14f)
        );
        color *= 1f + wave * (hardGlass ? 0.012f : 0.025f);

        if (v > 0.31f && v < 0.60f)
        {
            color = Color.Lerp(spec.Secondary, spec.Accent, 0.62f);
            if (IsPseudoPrint(u, v, spec.Seed, 8f, 18f))
            {
                color = Color.Lerp(color, spec.Primary, 0.86f);
            }
        }
        return color;
    }

    private static Color BuildGlass(
        MaterialSpec spec,
        float u,
        float v,
        float wave,
        float noise
    )
    {
        float verticalWave = 0.5f + 0.5f * Mathf.Sin(
            u * Mathf.PI * 10f + v * Mathf.PI * 2f + spec.Seed
        );
        float rim = Mathf.SmoothStep(0.86f, 1f, v);
        Color color = Color.Lerp(
            spec.Primary,
            spec.Secondary,
            0.08f + verticalWave * 0.12f + rim * 0.20f
        );
        color = Color.Lerp(color, spec.Accent, Mathf.Abs(wave) * 0.05f);
        return color * (1f + noise * 0.08f);
    }

    private static bool IsPseudoPrint(
        float u,
        float v,
        int seed,
        float rows,
        float columns
    )
    {
        float row = Mathf.Repeat(v * rows + seed * 0.013f, 1f);
        float column = Mathf.Repeat(
            u * columns + Mathf.Floor(v * rows) * 0.31f + seed * 0.017f,
            1f
        );
        return row > 0.28f && row < 0.38f &&
               column < 0.58f &&
               u > 0.12f && u < 0.88f &&
               v > 0.18f && v < 0.70f;
    }

    private static void ConfigureMaterial(
        Material material,
        Texture2D texture,
        MaterialSpec spec
    )
    {
        SetTexture(material, "_BaseColorMap", texture);
        SetTexture(material, "_MainTex", texture);
        SetFloat(material, "_Metallic", 0f);
        SetFloat(material, "_Smoothness", spec.Smoothness);
        SetFloat(material, "_CoatMask", spec.CoatMask);
        Color baseColor = new Color(1f, 1f, 1f, spec.Alpha);
        SetColor(material, "_BaseColor", baseColor);
        SetColor(material, "_Color", baseColor);

        if (spec.Transparent)
        {
            ConfigureTransparentGlass(material);
        }
        else
        {
            ConfigureOpaque(material);
        }
    }

    private static void ConfigureOpaque(Material material)
    {
        SetFloat(material, "_SurfaceType", 0f);
        SetFloat(material, "_ZWrite", 1f);
        SetFloat(material, "_DoubleSidedEnable", 0f);
        material.SetOverrideTag("RenderType", string.Empty);
        material.DisableKeyword("_SURFACE_TYPE_TRANSPARENT");
        material.renderQueue = -1;
        material.doubleSidedGI = false;
    }

    private static void ConfigureTransparentGlass(Material material)
    {
        SetFloat(material, "_SurfaceType", 1f);
        SetFloat(material, "_BlendMode", 0f);
        SetFloat(material, "_SrcBlend", 1f);
        SetFloat(material, "_DstBlend", 10f);
        SetFloat(material, "_AlphaSrcBlend", 1f);
        SetFloat(material, "_AlphaDstBlend", 10f);
        SetFloat(material, "_ZWrite", 0f);
        SetFloat(material, "_EnableFogOnTransparent", 1f);
        SetFloat(material, "_EnableBlendModePreserveSpecularLighting", 1f);
        SetFloat(material, "_DoubleSidedEnable", 1f);
        SetFloat(material, "_CullMode", 0f);
        SetFloat(material, "_CullModeForward", 0f);
        material.SetOverrideTag("RenderType", "Transparent");
        material.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
        material.EnableKeyword("_ENABLE_FOG_ON_TRANSPARENT");
        material.renderQueue = 3000;
        material.doubleSidedGI = true;
    }

    private static void SetTexture(Material material, string property, Texture texture)
    {
        if (material.HasProperty(property))
        {
            material.SetTexture(property, texture);
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

    private static void EnsureFolder(string folderPath)
    {
        if (AssetDatabase.IsValidFolder(folderPath))
        {
            return;
        }
        string parent = Path.GetDirectoryName(folderPath).Replace("\\", "/");
        EnsureFolder(parent);
        AssetDatabase.CreateFolder(parent, Path.GetFileName(folderPath));
    }
}
