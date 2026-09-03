using System.IO;
using UnityEditor;
using UnityEngine;

public static class GeneratedWallMaterialCreator
{
    private const string TextureFolder =
        "Assets/_MastersProject/Materials/Textures/Walls/Generated";

    private const string MaterialFolder =
        "Assets/_MastersProject/Materials/Walls/Generated";

    [MenuItem("MastersProject/Materials/Create Generated Wall Materials")]
    public static void CreateGeneratedWallMaterials()
    {
        EnsureFolder("Assets/_MastersProject/Materials");
        EnsureFolder("Assets/_MastersProject/Materials/Walls");
        EnsureFolder(MaterialFolder);

        string[] textureGuids = AssetDatabase.FindAssets("t:Texture2D Wall_", new[] { TextureFolder });

        if (textureGuids.Length == 0)
        {
            Debug.LogWarning($"No generated wall textures found in {TextureFolder}.");
            return;
        }

        Shader shader = Shader.Find("HDRP/Lit");

        if (shader == null)
        {
            shader = Shader.Find("Universal Render Pipeline/Lit");
        }

        if (shader == null)
        {
            shader = Shader.Find("Standard");
        }

        if (shader == null)
        {
            Debug.LogError("Could not find HDRP/Lit, URP/Lit, or Standard shader.");
            return;
        }

        int createdOrUpdated = 0;

        foreach (string guid in textureGuids)
        {
            string texturePath = AssetDatabase.GUIDToAssetPath(guid);

            if (Path.GetFileName(texturePath) == "Wall_Texture_ContactSheet.png")
            {
                continue;
            }

            Texture2D texture = AssetDatabase.LoadAssetAtPath<Texture2D>(texturePath);

            if (texture == null)
            {
                continue;
            }

            string textureName = Path.GetFileNameWithoutExtension(texturePath);
            string materialName = "M_" + textureName;
            string materialPath = $"{MaterialFolder}/{materialName}.mat";

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

            AssignBaseTexture(material, texture);
            AssignReasonableDefaults(material);

            EditorUtility.SetDirty(material);
            createdOrUpdated++;
        }

        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();

        Debug.Log(
            $"Created/updated {createdOrUpdated} generated wall materials in {MaterialFolder}. " +
            "Drag these materials into MaterialRandomizer > Wall Material Options."
        );
    }

    private static void AssignBaseTexture(Material material, Texture2D texture)
    {
        if (material.HasProperty("_BaseColorMap"))
        {
            material.SetTexture("_BaseColorMap", texture);
            material.SetTextureScale("_BaseColorMap", new Vector2(2f, 2f));
        }

        if (material.HasProperty("_MainTex"))
        {
            material.SetTexture("_MainTex", texture);
            material.SetTextureScale("_MainTex", new Vector2(2f, 2f));
        }
    }

    private static void AssignReasonableDefaults(Material material)
    {
        if (material.HasProperty("_BaseColor"))
        {
            material.SetColor("_BaseColor", Color.white);
        }

        if (material.HasProperty("_Color"))
        {
            material.SetColor("_Color", Color.white);
        }

        if (material.HasProperty("_Smoothness"))
        {
            material.SetFloat("_Smoothness", 0.35f);
        }

        if (material.HasProperty("_Metallic"))
        {
            material.SetFloat("_Metallic", 0f);
        }
    }

    private static void EnsureFolder(string folderPath)
    {
        if (AssetDatabase.IsValidFolder(folderPath))
        {
            return;
        }

        string parent = Path.GetDirectoryName(folderPath).Replace("\\", "/");
        string folderName = Path.GetFileName(folderPath);

        EnsureFolder(parent);
        AssetDatabase.CreateFolder(parent, folderName);
    }
}
