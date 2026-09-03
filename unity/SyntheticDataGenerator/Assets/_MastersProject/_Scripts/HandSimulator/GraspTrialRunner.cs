using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GraspTrialRunner : MonoBehaviour
{
    [Header("References")]
    public GraspCandidateGenerator candidateGenerator;
    public HandProxyController handController;
    public StraightLineApproachController approachController;
    public GraspOutcomeEvaluator outcomeEvaluator;
    public Transform wristRoot;
    public Transform targetObject;

    [Header("Sequence")]
    public float closeDelay = 0.3f;
    public float liftHeight = 0.12f;
    public float liftSpeed = 0.10f;
    public float holdDuration = 1.0f;
    public float resetDelay = 0.5f;


    public GraspTrialJsonlWriter trialWriter;
    public string sceneId = "grasp_trial_single_object_test";
    public string frameId = "manual_trial";
    public string imagePath = "";


    [Header("Debug")]
    public bool runOnStart = true;

    private Vector3 initialObjectPosition;
    private Quaternion initialObjectRotation;
    private Rigidbody targetRigidbody;

    private IEnumerator Start()
    {
        if (!runOnStart)
        {
            yield break;
        }

        yield return RunAllTrials();
    }

    public IEnumerator RunAllTrials()
    {
        if (!ValidateReferences())
        {
            yield break;
        }

        initialObjectPosition = targetObject.position;
        initialObjectRotation = targetObject.rotation;
        targetRigidbody = targetObject.GetComponent<Rigidbody>();

        List<GraspCandidate> candidates = candidateGenerator.GenerateCandidates();

        Debug.Log($"Running {candidates.Count} grasp candidates.");

        foreach (GraspCandidate candidate in candidates)
        {
            yield return RunCandidateTrial(candidate);
        }

        Debug.Log("All grasp candidate trials complete.");
    }

    private IEnumerator RunCandidateTrial(GraspCandidate candidate)
    {
        ResetObject();
        handController.ResetHand();

        yield return new WaitForSeconds(resetDelay);

        approachController.targetObject = targetObject;
        outcomeEvaluator.BeginTrial(targetObject);

        handController.SetAperture(candidate.handAperture);
        handController.SetWristRoll(candidate.wristRollDegrees);

        approachController.PlaceAtStartPose();

        yield return approachController.MoveToClosurePose();

        yield return new WaitForSeconds(closeDelay);

        handController.Close();

        yield return new WaitForSeconds(closeDelay);

        Vector3 liftTarget = wristRoot.position + Vector3.up * liftHeight;

        while (Vector3.Distance(wristRoot.position, liftTarget) > 0.005f)
        {
            wristRoot.position = Vector3.MoveTowards(
                wristRoot.position,
                liftTarget,
                liftSpeed * Time.deltaTime
            );

            yield return null;
        }

        yield return new WaitForSeconds(holdDuration);

        GraspOutcome outcome = outcomeEvaluator.Evaluate();

        if (trialWriter != null)
        {
            trialWriter.WriteTrial(
                sceneId,
                frameId,
                imagePath,
                targetObject,
                candidate,
                outcome
            );
        }

        Debug.Log(
            $"Candidate {candidate.candidateId}: " +
            $"success={outcome.success}, " +
            $"score={outcome.successScore}, " +
            $"reason={outcome.reason}, " +
            $"liftHeight={outcome.liftHeight:F3}"
        );
    }

    private void ResetObject()
    {
        if (targetObject == null)
        {
            return;
        }

        targetObject.SetParent(null, true);
        targetObject.position = initialObjectPosition;
        targetObject.rotation = initialObjectRotation;

        if (targetRigidbody != null)
        {
            targetRigidbody.isKinematic = false;
            targetRigidbody.linearVelocity = Vector3.zero;
            targetRigidbody.angularVelocity = Vector3.zero;
        }
    }

    private bool ValidateReferences()
    {
        if (candidateGenerator == null ||
            handController == null ||
            approachController == null ||
            outcomeEvaluator == null ||
            wristRoot == null ||
            targetObject == null)
        {
            Debug.LogError("GraspTrialRunner references are missing.");
            return false;
        }

        return true;
    }
}