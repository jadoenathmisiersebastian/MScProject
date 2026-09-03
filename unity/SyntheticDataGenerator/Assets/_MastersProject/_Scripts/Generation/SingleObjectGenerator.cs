using System.Collections;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using System.Collections.Generic;

public class SingleObjectTabletopGenerator : MonoBehaviour
{
    [Header("Object Library")]
    public PerceptionObjectLibrary objectLibrary;
    public Transform spawnedObjectsParent;

    [Header("Placement")]
    public Transform tableSurface;
    public TabletopPlacementValidator placementValidator;
    public ObjectKillPlane objectKillPlane;
    public Vector2 xRange = new Vector2(-0.45f, 0.45f);
    public Vector2 zRange = new Vector2(-0.25f, 0.25f);
    public float objectSpawnHeight = 0.2f;

    [Header("Capture")]
    public PerceptionCamera perceptionCamera;

    [Header("Spawn Mode")]
    public TargetObjectSpawnMode targetObjectSpawnMode = TargetObjectSpawnMode.DropFromHeight;
    public float tablePlacementClearance = 0.01f;

    [Header("Fallback Resting Pose")]
    public bool useDiscreteRestingPoses = true;
    [Tooltip("Used only when the selected prefab has no TargetObjectGenerationProfile. Weights: upright, upside-down, side Z +90, side Z -90, side X +90, side X -90.")]
    public float[] restingPoseWeights = { 4f, 1f, 1f, 1f, 1f, 1f };

    [Header("Generation")]
    public int sampleCount = 20;
    public int maxAttemptsPerSample = 10;

    [Header("Negative Samples")]
    [Range(0f, 1f)]
    [Tooltip("Probability that a captured frame contains no target object.")]
    public float nullSampleProbability = 0.1f;

    [Header("Random Seed")]
    public bool useFixedSeed = true;
    public int randomSeed = 12345;

    [Header("Settling")]
    public float initialPhysicsDelay = 0.5f;
    public float maxSettleWaitTime = 3.0f;
    public float settleCheckInterval = 0.05f;
    public float maxVelocity = 0.005f;
    public float maxAngularVelocity = 0.02f;

    [Header("Randomizers")]
    public TabletopCameraRandomizer cameraRandomizer;
    public LightingRandomizer lightingRandomizer;
    public MaterialRandomizer materialRandomizer;
    public ObjectScaleRandomizer objectScaleRandomizer;
    public EnvironmentLayoutRandomizer environmentLayoutRandomizer;
    public FurnitureLayoutRandomizer furnitureLayoutRandomizer;

    [Header("Vision Labels")]
    public SyntheticVisionLabelExporter visionLabelExporter;
    public Camera captureCamera;
    public string sceneId = "single_object_tabletop";
    public string imagePathPrefix = "";

    [Header("Debug")]
    public float debugSampleDelay = 0.0f;

    private GameObject currentObject;
    private bool currentSampleIsNull;

    private IEnumerator Start()
    {
        if (useFixedSeed)
        {
            Random.InitState(randomSeed);
        }

        int generatedSamples = 0;
        int attempts = 0;

        while (generatedSamples < sampleCount)
        {
            if (attempts == 0)
            {
                currentSampleIsNull = Random.value < nullSampleProbability;
            }

            attempts++;

            if (objectKillPlane != null)
            {
                objectKillPlane.ResetState();
            }

            if (environmentLayoutRandomizer != null)
            {
                environmentLayoutRandomizer.RandomizeEnvironment();
            }

            if (furnitureLayoutRandomizer != null)
            {
                furnitureLayoutRandomizer.RandomizeFurnitureLayout();
            }

            GenerateSample(generatedSamples);

            if (cameraRandomizer != null)
            {
                cameraRandomizer.RandomizeCamera();
            }

            if (lightingRandomizer != null)
            {
                lightingRandomizer.RandomizeLighting();
            }

            if (materialRandomizer != null)
            {
                materialRandomizer.RandomizeMaterials();
            }

            yield return new WaitForSeconds(initialPhysicsDelay);
            yield return WaitForObjectToSettle();

            bool validPlacement = IsCurrentSampleValid(out string rejectionReason);

            if (!validPlacement)
            {
                Debug.LogWarning(
                    $"Rejected sample {generatedSamples:0000}. " +
                    $"Attempt {attempts}/{maxAttemptsPerSample}. " +
                    $"Reason: {rejectionReason}"
                );

                CleanupSample();

                if (attempts >= maxAttemptsPerSample)
                {
                    Debug.LogError(
                        $"Failed to generate valid sample {generatedSamples:0000} " +
                        $"after {maxAttemptsPerSample} attempts."
                    );

                    attempts = 0;
                    generatedSamples++;
                }

                continue;
            }

            if (perceptionCamera != null)
            {
                perceptionCamera.RequestCapture();
                
            }
            else
            {
                Debug.LogError("Perception Camera is not assigned.");
            }

            if (visionLabelExporter != null)
            {
                visionLabelExporter.WriteFrame(
                    generatedSamples,
                    sceneId,
                    $"{imagePathPrefix}step{generatedSamples}.camera.png",
                    captureCamera,
                    tableSurface,
                    currentSampleIsNull
                        ? new List<GameObject>()
                        : new List<GameObject> { currentObject },
                    currentSampleIsNull
                );
            }

            yield return new WaitForEndOfFrame();

            if (debugSampleDelay > 0f)
            {
                yield return new WaitForSeconds(debugSampleDelay);
            }

            CleanupSample();

            generatedSamples++;
            attempts = 0;
        }

        Debug.Log($"Finished generating {sampleCount} samples.");
    }

    private void GenerateSample(int sampleIndex)
    {
        CleanupSample();

        if (objectLibrary == null)
        {
            Debug.LogError("Object library is not assigned.");
            return;
        }

        if (tableSurface == null)
        {
            Debug.LogError("Table surface is not assigned.");
            return;
        }

        if (spawnedObjectsParent == null)
        {
            Debug.LogError("Spawned objects parent is not assigned.");
            return;
        }

        if (currentSampleIsNull)
        {
            Debug.Log($"Sample {sampleIndex:0000} is a null target sample.");
            return;
        }

        GameObject selectedPrefab = objectLibrary.GetRandomPrefab();

        if (selectedPrefab == null)
        {
            return;
        }

        Vector3 localPosition = new Vector3(
            Random.Range(xRange.x, xRange.y),
            objectSpawnHeight,
            Random.Range(zRange.x, zRange.y)
        );

        Vector3 worldPosition = tableSurface.TransformPoint(localPosition);

        float[] selectedPoseWeights = objectLibrary.GetRestingPoseWeights(
            selectedPrefab,
            restingPoseWeights
        );

        Quaternion rotation = useDiscreteRestingPoses
            ? TargetObjectRestingPoseUtility.GetRandomDiscreteRestingPoseWithYaw(selectedPoseWeights)
            : TargetObjectRestingPoseUtility.GetRandomYawOnlyPose();

        currentObject = Instantiate(
            selectedPrefab,
            worldPosition,
            rotation,
            spawnedObjectsParent
        );

        if (objectScaleRandomizer != null)
        {
            objectScaleRandomizer.RandomizeScale(currentObject);
        }


        if (targetObjectSpawnMode == TargetObjectSpawnMode.PlaceOnSurface)
        {
            float tableWorldY = tableSurface.position.y + tablePlacementClearance;
            TargetObjectRestingPoseUtility.AlignRendererBoundsBottomToWorldY(
                currentObject,
                tableWorldY
            );
            TargetObjectRestingPoseUtility.ResetRigidbodyVelocities(currentObject);
        }

        currentObject.name = $"SampleObject_{sampleIndex:0000}_{selectedPrefab.name}";
    }

    private bool IsCurrentSampleValid(out string reason)
    {
        reason = null;

        if (currentObject == null)
        {
            if (currentSampleIsNull)
            {
                return true;
            }

            reason = "Current object is null for a positive sample.";
            return false;
        }

        if (objectKillPlane != null && objectKillPlane.WasTriggered)
        {
            reason = "Object kill plane was triggered.";
            return false;
        }

        if (placementValidator == null)
        {
            Debug.LogWarning("No placement validator assigned. Accepting sample by default.");
            return true;
        }

        reason = placementValidator.GetPlacementInvalidReason(currentObject);
        return reason == null;
    }

    private IEnumerator WaitForObjectToSettle()
    {
        float elapsed = 0f;

        while (elapsed < maxSettleWaitTime)
        {
            if (placementValidator == null || currentObject == null)
            {
                yield break;
            }

            if (placementValidator.IsObjectSettled(
                currentObject,
                maxVelocity,
                maxAngularVelocity
            ))
            {
                yield break;
            }

            elapsed += settleCheckInterval;
            yield return new WaitForSeconds(settleCheckInterval);
        }

        Debug.LogWarning("Object did not fully settle before capture.");
    }

    private void CleanupSample()
    {
        if (currentObject != null)
        {
            Destroy(currentObject);
            currentObject = null;
        }
    }
}
