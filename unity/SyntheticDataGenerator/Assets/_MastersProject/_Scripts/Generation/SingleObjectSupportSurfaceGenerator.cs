using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;

public class SingleObjectSupportSurfaceGenerator : MonoBehaviour
{
    [Header("Object Library")]
    public PerceptionObjectLibrary objectLibrary;
    public Transform spawnedObjectsParent;

    [Header("Support Surfaces")]
    public SupportSurfaceRegistry supportSurfaceRegistry;

    [Header("Validation")]
    public ObjectKillPlane objectKillPlane;

    [Header("Capture")]
    public PerceptionCamera perceptionCamera;

    [Header("Generation")]
    public int sampleCount = 20;
    public int maxAttemptsPerSample = 10;
    [Tooltip("Abort generation after this many complete failed retry cycles without a successful capture.")]
    public int maxConsecutiveFailedSampleCycles = 100;

    [Header("Negative Samples")]
    [Range(0f, 1f)]
    [Tooltip("Probability that a captured frame contains no target object.")]
    public float nullSampleProbability = 0.1f;

    [Header("Random Seed")]
    public bool useFixedSeed = false;
    public int randomSeed = 12345;

    [Header("Spawn Mode")]
    public TargetObjectSpawnMode targetObjectSpawnMode = TargetObjectSpawnMode.DropFromHeight;
    public float surfacePlacementClearance = 0.01f;

    [Header("Fallback Resting Pose")]
    public bool useDiscreteRestingPoses = true;
    [Tooltip("Used only when the selected prefab has no TargetObjectGenerationProfile. Weights: upright, upside-down, side Z +90, side Z -90, side X +90, side X -90.")]
    public float[] restingPoseWeights = { 4f, 1f, 1f, 1f, 1f, 1f };

    [Header("Settling")]
    public float initialPhysicsDelay = 0.5f;
    public float maxSettleWaitTime = 3.0f;
    public float settleCheckInterval = 0.05f;
    public float maxVelocity = 0.005f;
    public float maxAngularVelocity = 0.02f;

    [Header("Randomizers")]
    public EnvironmentLayoutRandomizer environmentLayoutRandomizer;
    public FurnitureLayoutRandomizer furnitureLayoutRandomizer;
    public WallFeatureRandomizer wallFeatureRandomizer;
    public SurfaceDistractorRandomizer surfaceDistractorRandomizer;
    public TabletopCameraRandomizer cameraRandomizer;
    public LightingRandomizer lightingRandomizer;
    public MaterialRandomizer materialRandomizer;
    public ObjectScaleRandomizer objectScaleRandomizer;

    [Header("Vision Labels")]
    public SyntheticVisionLabelExporter visionLabelExporter;
    public Camera captureCamera;
    public string sceneId = "single_object_support_surface";
    public string imagePathPrefix = "";

    [Header("Debug")]
    public float debugSampleDelay = 0.0f;
    public bool logSelectedSurface = true;

    private GameObject currentObject;
    private GameObject selectedPrefabForSample;
    private SupportSurface selectedSurface;
    private bool currentSampleIsNull;
    private bool hasSelectedSampleDefinition;
    private int consecutiveFailedSampleCycles;

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
            if (!hasSelectedSampleDefinition)
            {
                currentSampleIsNull = Random.value < nullSampleProbability;
                selectedPrefabForSample = currentSampleIsNull
                    ? null
                    : objectLibrary.GetRandomPrefab();
                hasSelectedSampleDefinition = true;

                if (!currentSampleIsNull && selectedPrefabForSample == null)
                {
                    throw new System.InvalidOperationException(
                        "Could not select a target prefab for a positive sample."
                    );
                }
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

            if (wallFeatureRandomizer != null)
            {
                wallFeatureRandomizer.RandomizeWallFeatures();
            }

            bool generated = GenerateSample(generatedSamples);

            if (!generated)
            {
                CleanupSample();
                RejectSample(generatedSamples, attempts, "Could not generate support-surface sample.", ref attempts);
                continue;
            }

            if (surfaceDistractorRandomizer != null)
            {
                surfaceDistractorRandomizer.RandomizeSurfaceDistractors(
                    supportSurfaceRegistry,
                    currentSampleIsNull ? null : selectedSurface
                );
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

            if (cameraRandomizer == null)
            {
                CleanupSample();
                RejectSample(
                    generatedSamples,
                    attempts,
                    "Camera randomizer is not assigned.",
                    ref attempts
                );
                continue;
            }

            string cameraRejectionReason;
            bool cameraPoseGenerated;

            if (currentSampleIsNull)
            {
                cameraPoseGenerated = cameraRandomizer.TryRandomizeNullCamera(
                    out cameraRejectionReason
                );
            }
            else
            {
                cameraPoseGenerated = cameraRandomizer.TryRandomizeCamera(
                    currentObject,
                    captureCamera,
                    out cameraRejectionReason
                );
            }

            if (!cameraPoseGenerated)
            {
                CleanupSample();
                RejectSample(generatedSamples, attempts, cameraRejectionReason, ref attempts);
                continue;
            }

            bool validPlacement = IsCurrentSampleValid(out string rejectionReason);

            if (!validPlacement)
            {
                CleanupSample();
                RejectSample(generatedSamples, attempts, rejectionReason, ref attempts);
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
                    selectedSurface != null ? selectedSurface.transform : null,
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
            selectedPrefabForSample = null;
            hasSelectedSampleDefinition = false;
            consecutiveFailedSampleCycles = 0;
        }

        Debug.Log($"Finished generating {sampleCount} support-surface samples.");
    }

    private bool GenerateSample(int sampleIndex)
    {
        CleanupSample();

        if (objectLibrary == null)
        {
            Debug.LogError("Object library is not assigned.");
            return false;
        }

        if (spawnedObjectsParent == null)
        {
            Debug.LogError("Spawned objects parent is not assigned.");
            return false;
        }

        if (supportSurfaceRegistry == null)
        {
            Debug.LogError("Support surface registry is not assigned.");
            return false;
        }

        if (currentSampleIsNull)
        {
            selectedSurface = null;

            if (logSelectedSurface)
            {
                Debug.Log(
                    $"Sample {sampleIndex:0000} is a null target sample. " +
                    "The camera will sample an independent room view."
                );
            }

            return true;
        }

        selectedSurface = supportSurfaceRegistry.GetRandomSurface();

        if (selectedSurface == null)
        {
            return false;
        }

        GameObject selectedPrefab = selectedPrefabForSample;

        if (selectedPrefab == null)
        {
            return false;
        }

        Vector3 spawnPosition = selectedSurface.SampleSpawnPosition();

        float[] selectedPoseWeights = objectLibrary.GetRestingPoseWeights(
            selectedPrefab,
            restingPoseWeights
        );

        Quaternion rotation = useDiscreteRestingPoses
            ? TargetObjectRestingPoseUtility.GetRandomDiscreteRestingPoseWithYaw(selectedPoseWeights)
            : TargetObjectRestingPoseUtility.GetRandomYawOnlyPose();

        currentObject = Instantiate(
            selectedPrefab,
            spawnPosition,
            rotation,
            spawnedObjectsParent
        );

        if (objectScaleRandomizer != null)
        {
            objectScaleRandomizer.RandomizeScale(currentObject);
        }


        if (targetObjectSpawnMode == TargetObjectSpawnMode.PlaceOnSurface)
        {
            float surfaceWorldY = selectedSurface.transform.position.y + surfacePlacementClearance;
            TargetObjectRestingPoseUtility.AlignRendererBoundsBottomToWorldY(
                currentObject,
                surfaceWorldY
            );
            TargetObjectRestingPoseUtility.ResetRigidbodyVelocities(currentObject);
        }

        currentObject.name =
            $"SampleObject_{sampleIndex:0000}_{selectedSurface.surfaceId}_{selectedPrefab.name}";

        if (logSelectedSurface)
        {
            Debug.Log(
                $"Sample {sampleIndex:0000} selected support surface: " +
                $"{selectedSurface.surfaceId}"
            );
        }

        return true;
    }

    private bool IsCurrentSampleValid(out string reason)
    {
        reason = null;

        if (currentObject == null)
        {
            if (!currentSampleIsNull)
            {
                reason = "Current object is null for a positive sample.";
                return false;
            }

            if (cameraRandomizer == null)
            {
                reason = "Camera randomizer is not assigned.";
                return false;
            }

            if (!cameraRandomizer.IsCurrentCameraPoseClear(out string cameraReason))
            {
                reason = $"Null-sample camera pose is invalid. {cameraReason}";
                return false;
            }

            return true;
        }

        if (selectedSurface == null)
        {
            reason = "Selected support surface is null.";
            return false;
        }

        if (objectKillPlane != null && objectKillPlane.WasTriggered)
        {
            reason = "Object kill plane was triggered.";
            return false;
        }

        if (!selectedSurface.IsObjectOnSurface(currentObject))
        {
            reason = $"Object is not on selected support surface {selectedSurface.surfaceId}.";
            return false;
        }

        if (!IsObjectSettled(currentObject, maxVelocity, maxAngularVelocity))
        {
            reason = "Object is not settled.";
            return false;
        }

        if (cameraRandomizer != null)
        {
            Vector3 cameraTargetPosition = GetCurrentObjectFocusPosition();

            if (!cameraRandomizer.IsCurrentCameraDistanceValid(cameraTargetPosition, out float cameraDistance))
            {
                reason = $"Camera distance {cameraDistance:F2}m is outside the allowed range.";
                return false;
            }

            if (!cameraRandomizer.IsTargetFullyVisible(
                currentObject,
                captureCamera,
                out string visibilityReason
            ))
            {
                reason = visibilityReason;
                return false;
            }
        }
        else
        {
            reason = "Camera randomizer is not assigned.";
            return false;
        }

        return true;
    }

    private IEnumerator WaitForObjectToSettle()
    {
        float elapsed = 0f;

        while (elapsed < maxSettleWaitTime)
        {
            if (currentObject == null)
            {
                yield break;
            }

            if (IsObjectSettled(currentObject, maxVelocity, maxAngularVelocity))
            {
                yield break;
            }

            elapsed += settleCheckInterval;
            yield return new WaitForSeconds(settleCheckInterval);
        }

        Debug.LogWarning("Object did not fully settle before capture.");
    }

    private Vector3 GetCurrentObjectFocusPosition()
    {
        if (currentObject == null)
        {
            if (selectedSurface != null)
            {
                return selectedSurface.GetFocusPosition();
            }

            return Vector3.zero;
        }

        Renderer[] renderers = currentObject.GetComponentsInChildren<Renderer>();

        if (renderers.Length == 0)
        {
            return currentObject.transform.position;
        }

        Bounds bounds = renderers[0].bounds;

        for (int i = 1; i < renderers.Length; i++)
        {
            bounds.Encapsulate(renderers[i].bounds);
        }

        return bounds.center;
    }

    private bool IsObjectSettled(GameObject obj, float maxAllowedVelocity, float maxAllowedAngularVelocity)
    {
        Rigidbody[] rigidbodies = obj.GetComponentsInChildren<Rigidbody>();

        if (rigidbodies.Length == 0)
        {
            return true;
        }

        foreach (Rigidbody rb in rigidbodies)
        {
            if (rb.linearVelocity.magnitude > maxAllowedVelocity)
            {
                return false;
            }

            if (rb.angularVelocity.magnitude > maxAllowedAngularVelocity)
            {
                return false;
            }
        }

        return true;
    }

    private void RejectSample(
        int sampleIndex,
        int attempts,
        string reason,
        ref int attemptCounter
    )
    {
        Debug.LogWarning(
            $"Rejected sample {sampleIndex:0000}. " +
            $"Attempt {attempts}/{maxAttemptsPerSample}. " +
            $"Reason: {reason}"
        );

        if (attempts >= maxAttemptsPerSample)
        {
            consecutiveFailedSampleCycles++;

            Debug.LogError(
                $"Failed retry cycle {consecutiveFailedSampleCycles}/" +
                $"{maxConsecutiveFailedSampleCycles} for sample " +
                $"{sampleIndex:0000} after {maxAttemptsPerSample} attempts. " +
                "The sample index will be retried."
            );

            if (consecutiveFailedSampleCycles >= maxConsecutiveFailedSampleCycles)
            {
                throw new System.InvalidOperationException(
                    $"Aborting generation after {consecutiveFailedSampleCycles} " +
                    $"failed cycles for sample {sampleIndex:0000}."
                );
            }

            attemptCounter = 0;
        }
    }

    private void CleanupSample()
    {
        if (currentObject != null)
        {
            Destroy(currentObject);
            currentObject = null;
        }

        selectedSurface = null;
    }
}
