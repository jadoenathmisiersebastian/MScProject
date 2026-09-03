using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Perception.GroundTruth.LabelManagement;

public class SyntheticVisionLabelExporter : MonoBehaviour
{
    [Header("Semantic Labels")]
    public IdLabelConfig idLabelConfig;

    [Header("Dataset Root")]
    [Tooltip("Output directory. Relative paths are resolved from the Unity project root.")]
    public string datasetRoot = "../../python/datasets/vision_raw/generated";

    public string labelsSubdirectory = "labels";
    public string outputFileName = "vision_labels.jsonl";

    private string outputPath;

    private void Awake()
    {
        string labelsDirectory = Path.Combine(datasetRoot, labelsSubdirectory);

        Directory.CreateDirectory(labelsDirectory);

        outputPath = Path.Combine(labelsDirectory, outputFileName);

        if (File.Exists(outputPath))
        {
            File.Delete(outputPath);
        }

        Debug.Log($"Writing synthetic vision labels to: {outputPath}");
    }

    public void WriteFrame(
        int frameIndex,
        string sceneId,
        string imagePath,
        Camera captureCamera,
        Transform tableSurface,
        List<GameObject> objects,
        bool isNullSample = false
    )
    {
        if (captureCamera == null)
        {
            Debug.LogWarning("Cannot write vision labels: capture camera is missing.");
            return;
        }

        VisionFrameRecord record = new VisionFrameRecord
        {
            frame_id = $"frame_{frameIndex:000000}",
            scene_id = sceneId,
            image_path = imagePath,
            camera = BuildCameraData(captureCamera),
            objects = BuildObjectData(captureCamera, tableSurface, objects),
            is_null_sample = isNullSample
        };

        if (record.is_null_sample && record.objects.Count > 0)
        {
            throw new System.InvalidOperationException(
                "A null sample cannot contain target objects."
            );
        }

        MarkFocusedObject(record);

        string json = JsonUtility.ToJson(record);
        File.AppendAllText(outputPath, json + "\n");
    }

    private VisionCameraData BuildCameraData(Camera camera)
    {
        return new VisionCameraData
        {
            position_world = ToArray(camera.transform.position),
            rotation_world_quat = ToArray(camera.transform.rotation),
            image_width = camera.pixelWidth,
            image_height = camera.pixelHeight,
            field_of_view_degrees = camera.fieldOfView
        };
    }

    private List<VisionObjectData> BuildObjectData(
        Camera camera,
        Transform tableSurface,
        List<GameObject> objects
    )
    {
        List<VisionObjectData> output = new List<VisionObjectData>();

        foreach (GameObject obj in objects)
        {
            if (obj == null)
            {
                continue;
            }

            Bounds bounds = GetObjectBounds(obj);
            Vector3 objectCenterWorld = bounds.center;
            Vector3 objectCenterCamera = camera.transform.InverseTransformPoint(objectCenterWorld);

            bool visibleInFrontOfCamera = objectCenterCamera.z > 0f;

            ProjectedBoundingBox projectedBox = ProjectBoundsToImage(camera, bounds);

            VisionObjectData objectData = new VisionObjectData
            {
                object_id = obj.GetInstanceID(),
                object_name = obj.name,
                class_name = CleanClassName(obj.name),
                semantic_class = GetSemanticClass(obj),

                bbox_xyxy = projectedBox.bbox_xyxy,
                image_center = projectedBox.image_center,
                normalized_center = projectedBox.normalized_center,
                bbox_area_pixels = projectedBox.area_pixels,
                bbox_area_normalized = projectedBox.area_normalized,

                position_world = ToArray(objectCenterWorld),
                position_camera = ToArray(objectCenterCamera),
                distance_camera_m = Vector3.Distance(camera.transform.position, objectCenterWorld),

                position_table = tableSurface != null
                    ? ToArray(tableSurface.InverseTransformPoint(objectCenterWorld))
                    : ToArray(objectCenterWorld),

                dimensions_m = ToArray(bounds.size),

                is_in_front_of_camera = visibleInFrontOfCamera,
                focus_distance = CalculateFocusDistance(projectedBox.normalized_center),
                is_focused_object = false
            };

            output.Add(objectData);
        }

        return output;
    }

    private string GetSemanticClass(GameObject obj)
    {
        if (idLabelConfig == null)
        {
            throw new System.InvalidOperationException(
                "SyntheticVisionLabelExporter requires an IdLabelConfig. Assign SS_IdLabelConfig in the Inspector."
            );
        }

        Labeling[] labelings = obj.GetComponentsInChildren<Labeling>(true);

        foreach (Labeling labeling in labelings)
        {
            if (idLabelConfig.TryGetMatchingConfigurationEntry(labeling, out IdLabelEntry entry))
            {
                return entry.label;
            }
        }

        throw new System.InvalidOperationException(
            $"Target object '{obj.name}' has no Labeling entry matching '{idLabelConfig.name}'."
        );
    }

    private void MarkFocusedObject(VisionFrameRecord record)
    {
        VisionObjectData bestObject = null;
        float bestFocusDistance = float.PositiveInfinity;

        foreach (VisionObjectData obj in record.objects)
        {
            if (!obj.is_in_front_of_camera)
            {
                continue;
            }

            if (obj.bbox_area_pixels <= 0f)
            {
                continue;
            }

            if (obj.focus_distance < bestFocusDistance)
            {
                bestFocusDistance = obj.focus_distance;
                bestObject = obj;
            }
        }

        if (bestObject != null)
        {
            bestObject.is_focused_object = true;
        }
    }

    private ProjectedBoundingBox ProjectBoundsToImage(Camera camera, Bounds bounds)
    {
        Vector3[] corners =
        {
            new Vector3(bounds.min.x, bounds.min.y, bounds.min.z),
            new Vector3(bounds.min.x, bounds.min.y, bounds.max.z),
            new Vector3(bounds.min.x, bounds.max.y, bounds.min.z),
            new Vector3(bounds.min.x, bounds.max.y, bounds.max.z),
            new Vector3(bounds.max.x, bounds.min.y, bounds.min.z),
            new Vector3(bounds.max.x, bounds.min.y, bounds.max.z),
            new Vector3(bounds.max.x, bounds.max.y, bounds.min.z),
            new Vector3(bounds.max.x, bounds.max.y, bounds.max.z)
        };

        float minX = float.PositiveInfinity;
        float minY = float.PositiveInfinity;
        float maxX = float.NegativeInfinity;
        float maxY = float.NegativeInfinity;

        bool hasPointInFront = false;

        foreach (Vector3 corner in corners)
        {
            Vector3 screenPoint = camera.WorldToScreenPoint(corner);

            if (screenPoint.z <= 0f)
            {
                continue;
            }

            hasPointInFront = true;

            minX = Mathf.Min(minX, screenPoint.x);
            maxX = Mathf.Max(maxX, screenPoint.x);
            minY = Mathf.Min(minY, screenPoint.y);
            maxY = Mathf.Max(maxY, screenPoint.y);
        }

        int imageWidth = camera.pixelWidth;
        int imageHeight = camera.pixelHeight;

        if (!hasPointInFront)
        {
            return ProjectedBoundingBox.Empty();
        }

        minX = Mathf.Clamp(minX, 0f, imageWidth);
        maxX = Mathf.Clamp(maxX, 0f, imageWidth);
        minY = Mathf.Clamp(minY, 0f, imageHeight);
        maxY = Mathf.Clamp(maxY, 0f, imageHeight);

        // Convert Unity bottom-left screen origin to top-left image origin.
        float x1 = minX;
        float y1 = imageHeight - maxY;
        float x2 = maxX;
        float y2 = imageHeight - minY;

        float width = Mathf.Max(0f, x2 - x1);
        float height = Mathf.Max(0f, y2 - y1);
        float areaPixels = width * height;

        return new ProjectedBoundingBox
        {
            bbox_xyxy = new float[] { x1, y1, x2, y2 },
            image_center = new float[] { x1 + width / 2f, y1 + height / 2f },
            normalized_center = new float[]
            {
                imageWidth > 0 ? (x1 + width / 2f) / imageWidth : 0f,
                imageHeight > 0 ? (y1 + height / 2f) / imageHeight : 0f
            },
            area_pixels = areaPixels,
            area_normalized = imageWidth > 0 && imageHeight > 0
                ? areaPixels / (imageWidth * imageHeight)
                : 0f
        };
    }

    private float CalculateFocusDistance(float[] normalizedCenter)
    {
        if (normalizedCenter == null || normalizedCenter.Length != 2)
        {
            return float.PositiveInfinity;
        }

        float dx = normalizedCenter[0] - 0.5f;
        float dy = normalizedCenter[1] - 0.5f;

        return Mathf.Sqrt(dx * dx + dy * dy);
    }

    private Bounds GetObjectBounds(GameObject obj)
    {
        Renderer[] renderers = obj.GetComponentsInChildren<Renderer>();

        if (renderers.Length > 0)
        {
            Bounds bounds = renderers[0].bounds;

            for (int i = 1; i < renderers.Length; i++)
            {
                bounds.Encapsulate(renderers[i].bounds);
            }

            return bounds;
        }

        Collider[] colliders = obj.GetComponentsInChildren<Collider>();

        if (colliders.Length > 0)
        {
            Bounds bounds = colliders[0].bounds;

            for (int i = 1; i < colliders.Length; i++)
            {
                bounds.Encapsulate(colliders[i].bounds);
            }

            return bounds;
        }

        return new Bounds(obj.transform.position, Vector3.zero);
    }

    private string CleanClassName(string objectName)
    {
        string cleaned = objectName;

        if (cleaned.StartsWith("SampleObject_"))
        {
            int lastUnderscore = cleaned.LastIndexOf('_');

            if (lastUnderscore >= 0 && lastUnderscore < cleaned.Length - 1)
            {
                cleaned = cleaned.Substring(lastUnderscore + 1);
            }
        }

        cleaned = cleaned.Replace("(Clone)", "");
        cleaned = cleaned.Trim();

        return cleaned;
    }

    private float[] ToArray(Vector3 value)
    {
        return new float[] { value.x, value.y, value.z };
    }

    private float[] ToArray(Quaternion value)
    {
        return new float[] { value.x, value.y, value.z, value.w };
    }
}

[System.Serializable]
public class VisionFrameRecord
{
    public string frame_id;
    public string scene_id;
    public string image_path;
    public VisionCameraData camera;
    public List<VisionObjectData> objects;
    public bool is_null_sample;
}

[System.Serializable]
public class VisionCameraData
{
    public float[] position_world;
    public float[] rotation_world_quat;
    public int image_width;
    public int image_height;
    public float field_of_view_degrees;
}

[System.Serializable]
public class VisionObjectData
{
    public int object_id;
    public string object_name;
    public string class_name;
    public string semantic_class;

    public float[] bbox_xyxy;
    public float[] image_center;
    public float[] normalized_center;
    public float bbox_area_pixels;
    public float bbox_area_normalized;

    public float[] position_world;
    public float[] position_camera;
    public float distance_camera_m;

    public float[] position_table;
    public float[] dimensions_m;

    public bool is_in_front_of_camera;
    public float focus_distance;
    public bool is_focused_object;
}

public class ProjectedBoundingBox
{
    public float[] bbox_xyxy;
    public float[] image_center;
    public float[] normalized_center;
    public float area_pixels;
    public float area_normalized;

    public static ProjectedBoundingBox Empty()
    {
        return new ProjectedBoundingBox
        {
            bbox_xyxy = new float[] { 0f, 0f, 0f, 0f },
            image_center = new float[] { 0f, 0f },
            normalized_center = new float[] { 0f, 0f },
            area_pixels = 0f,
            area_normalized = 0f
        };
    }
}
