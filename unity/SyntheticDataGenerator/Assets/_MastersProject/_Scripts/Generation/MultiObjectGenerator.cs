using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;

public class MultiObjectTabletopGenerator : MonoBehaviour
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
    public float minSpawnDistance = 0.18f;
    public int maxPlacementAttemptsPerObject = 30;

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
    public int minObjectsPerSample = 2;
    public int maxObjectsPerSample = 4;

    [Header("Negative Samples")]
    [Range(0f, 1f)]
    [Tooltip("Probability that a captured frame contains no target objects.")]
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

    [Header("Debug")]
    public float debugSampleDelay = 0.0f;

    [Header("Vision Labels")]
    public SyntheticVisionLabelExporter visionLabelExporter;
    public Camera captureCamera;
    public string sceneId = "multi_object_tabletop";
    public string imagePathPrefix = "";

    private readonly List<GameObject> currentObjects = new List<GameObject>();
    private readonly List<Vector3> sampledLocalPositions = new List<Vector3>();
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

            bool generated = GenerateSample(generatedSamples);

            if (!generated)
            {
                CleanupSample();

                Debug.LogWarning(
                    $"Rejected sample {generatedSamples:0000}. " +
                    $"Attempt {attempts}/{maxAttemptsPerSample}. " +
                    "Reason: Could not generate all object placements."
                );

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
            yield return WaitForObjectsToSettle();

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
                    currentObjects,
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

        Debug.Log($"Finished generating {sampleCount} multi-object samples.");
    }

    private bool GenerateSample(int sampleIndex)
    {
        CleanupSample();

        if (objectLibrary == null)
        {
            Debug.LogError("Object library is not assigned.");
            return false;
        }

        if (tableSurface == null)
        {
            Debug.LogError("Table surface is not assigned.");
            return false;
        }

        if (spawnedObjectsParent == null)
        {
            Debug.LogError("Spawned objects parent is not assigned.");
            return false;
        }

        if (currentSampleIsNull)
        {
            Debug.Log($"Sample {sampleIndex:0000} is a null target sample.");
            return true;
        }

        int objectCount = Random.Range(
            minObjectsPerSample,
            maxObjectsPerSample + 1
        );

        for (int i = 0; i < objectCount; i++)
        {
            GameObject selectedPrefab = objectLibrary.GetRandomPrefab();

            if (selectedPrefab == null)
            {
                return false;
            }

            if (!TrySampleLocalPosition(out Vector3 localPosition))
            {
                return false;
            }

            Vector3 worldPosition = tableSurface.TransformPoint(localPosition);

            float[] selectedPoseWeights = objectLibrary.GetRestingPoseWeights(
                selectedPrefab,
                restingPoseWeights
            );

            Quaternion rotation = useDiscreteRestingPoses
                ? TargetObjectRestingPoseUtility.GetRandomDiscreteRestingPoseWithYaw(selectedPoseWeights)
                : TargetObjectRestingPoseUtility.GetRandomYawOnlyPose();

            GameObject spawnedObject = Instantiate(
                selectedPrefab,
                worldPosition,
                rotation,
                spawnedObjectsParent
            );

            if (objectScaleRandomizer != null)
            {
                objectScaleRandomizer.RandomizeScale(spawnedObject);
            }


            if (targetObjectSpawnMode == TargetObjectSpawnMode.PlaceOnSurface)
            {
                float tableWorldY = tableSurface.position.y + tablePlacementClearance;
                TargetObjectRestingPoseUtility.AlignRendererBoundsBottomToWorldY(
                    spawnedObject,
                    tableWorldY
                );
                TargetObjectRestingPoseUtility.ResetRigidbodyVelocities(spawnedObject);
            }

            spawnedObject.name =
                $"SampleObject_{sampleIndex:0000}_{i:00}_{selectedPrefab.name}";

            currentObjects.Add(spawnedObject);
        }

        return true;
    }

    private bool TrySampleLocalPosition(out Vector3 localPosition)
    {
        for (int attempt = 0; attempt < maxPlacementAttemptsPerObject; attempt++)
        {
            localPosition = new Vector3(
                Random.Range(xRange.x, xRange.y),
                objectSpawnHeight,
                Random.Range(zRange.x, zRange.y)
            );

            if (IsFarEnoughFromExistingObjects(localPosition))
            {
                sampledLocalPositions.Add(localPosition);
                return true;
            }
        }

        localPosition = Vector3.zero;
        return false;
    }

    private bool IsFarEnoughFromExistingObjects(Vector3 candidateLocalPosition)
    {
        Vector2 candidate = new Vector2(
            candidateLocalPosition.x,
            candidateLocalPosition.z
        );

        foreach (Vector3 existingLocalPosition in sampledLocalPositions)
        {
            Vector2 existing = new Vector2(
                existingLocalPosition.x,
                existingLocalPosition.z
            );

            if (Vector2.Distance(candidate, existing) < minSpawnDistance)
            {
                return false;
            }
        }

        return true;
    }

    private bool IsCurrentSampleValid(out string reason)
    {
        reason = null;

        if (currentObjects.Count == 0)
        {
            if (currentSampleIsNull)
            {
                return true;
            }

            reason = "No objects were spawned for a positive sample.";
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

        for (int i = 0; i < currentObjects.Count; i++)
        {
            GameObject obj = currentObjects[i];

            if (obj == null)
            {
                reason = $"Object index {i} is null.";
                return false;
            }

            string invalidReason = placementValidator.GetPlacementInvalidReason(obj);

            if (invalidReason != null)
            {
                reason = $"{obj.name}: {invalidReason}";
                return false;
            }
        }

        return true;
    }

    private IEnumerator WaitForObjectsToSettle()
    {
        float elapsed = 0f;

        while (elapsed < maxSettleWaitTime)
        {
            if (placementValidator == null || currentObjects.Count == 0)
            {
                yield break;
            }

            bool allSettled = true;

            foreach (GameObject obj in currentObjects)
            {
                if (obj == null)
                {
                    allSettled = false;
                    break;
                }

                if (!placementValidator.IsObjectSettled(
                    obj,
                    maxVelocity,
                    maxAngularVelocity
                ))
                {
                    allSettled = false;
                    break;
                }
            }

            if (allSettled)
            {
                yield break;
            }

            elapsed += settleCheckInterval;
            yield return new WaitForSeconds(settleCheckInterval);
        }

        Debug.LogWarning("One or more objects did not fully settle before capture.");
    }

    private void CleanupSample()
    {
        foreach (GameObject obj in currentObjects)
        {
            if (obj != null)
            {
                Destroy(obj);
            }
        }

        currentObjects.Clear();
        sampledLocalPositions.Clear();
    }
}
