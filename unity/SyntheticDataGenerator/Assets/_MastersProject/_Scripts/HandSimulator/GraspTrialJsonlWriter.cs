using System.IO;
using UnityEngine;

public class GraspTrialJsonlWriter : MonoBehaviour
{
    [Header("Output")]
    public string outputDirectory = "../../python/datasets/grasp_trials";

    public string outputFileName = "grasp_trials_debug.jsonl";

    private string outputPath;
    private int trialCounter;

    private void Awake()
    {
        Directory.CreateDirectory(outputDirectory);

        outputPath = Path.Combine(outputDirectory, outputFileName);

        if (File.Exists(outputPath))
        {
            File.Delete(outputPath);
        }

        Debug.Log($"Writing grasp trials to: {outputPath}");
    }

    public void WriteTrial(
        string sceneId,
        string frameId,
        string imagePath,
        Transform targetObject,
        GraspCandidate candidate,
        GraspOutcome outcome
    )
    {
        GraspTrialRecord record = new GraspTrialRecord
        {
            trial_id = $"trial_{trialCounter:000000}",
            scene_id = sceneId,
            frame_id = frameId,
            image_path = imagePath,
            object_data = BuildObjectData(targetObject),
            candidate_grasp = BuildCandidateData(candidate),
            outcome = BuildOutcomeData(outcome)
        };

        string json = JsonUtility.ToJson(record);
        json = json.Replace("\"object_data\":", "\"object\":");

        File.AppendAllText(outputPath, json + "\n");

        trialCounter++;
    }

    private GraspTrialObjectData BuildObjectData(Transform targetObject)
    {
        Bounds bounds = GetObjectBounds(targetObject.gameObject);
        Vector3 center = bounds.center;
        Vector3 size = bounds.size;

        return new GraspTrialObjectData
        {
            class_name = targetObject.name,
            instance_id = targetObject.GetInstanceID(),
            bbox_xyxy = new float[] { 0f, 0f, 0f, 0f },
            image_center = new float[] { 0f, 0f },
            object_pose_camera = new GraspTrialPoseData
            {
                position = new float[] { center.x, center.y, center.z },
                rotation_quat = new float[]
                {
                    targetObject.rotation.x,
                    targetObject.rotation.y,
                    targetObject.rotation.z,
                    targetObject.rotation.w
                }
            },
            dimensions_m = new float[] { size.x, size.y, size.z }
        };
    }

    private GraspTrialCandidateData BuildCandidateData(GraspCandidate candidate)
    {
        return new GraspTrialCandidateData
        {
            candidate_id = candidate.candidateId,
            grasp_type = candidate.graspType,
            wrist_roll_degrees = candidate.wristRollDegrees,
            hand_aperture = candidate.handAperture,
            approach_direction_camera = new float[]
            {
                candidate.approachDirectionCamera.x,
                candidate.approachDirectionCamera.y,
                candidate.approachDirectionCamera.z
            }
        };
    }

    private GraspTrialOutcomeData BuildOutcomeData(GraspOutcome outcome)
    {
        return new GraspTrialOutcomeData
        {
            success = outcome.success,
            success_score = outcome.successScore,
            lift_height_m = outcome.liftHeight,
            hold_duration_s = 0f,
            slip_distance_m = 0f,
            pregrasp_collision = false
        };
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

        return new Bounds(obj.transform.position, Vector3.zero);
    }
}

[System.Serializable]
public class GraspTrialRecord
{
    public string trial_id;
    public string scene_id;
    public string frame_id;
    public string image_path;
    public GraspTrialObjectData object_data;
    public GraspTrialCandidateData candidate_grasp;
    public GraspTrialOutcomeData outcome;
}

[System.Serializable]
public class GraspTrialObjectData
{
    public string class_name;
    public int instance_id;
    public float[] bbox_xyxy;
    public float[] image_center;
    public GraspTrialPoseData object_pose_camera;
    public float[] dimensions_m;
}

[System.Serializable]
public class GraspTrialPoseData
{
    public float[] position;
    public float[] rotation_quat;
}

[System.Serializable]
public class GraspTrialCandidateData
{
    public string candidate_id;
    public string grasp_type;
    public float wrist_roll_degrees;
    public float hand_aperture;
    public float[] approach_direction_camera;
}

[System.Serializable]
public class GraspTrialOutcomeData
{
    public bool success;
    public float success_score;
    public float lift_height_m;
    public float hold_duration_s;
    public float slip_distance_m;
    public bool pregrasp_collision;
}
