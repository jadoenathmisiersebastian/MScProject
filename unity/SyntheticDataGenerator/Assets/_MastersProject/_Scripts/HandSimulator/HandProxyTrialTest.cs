using System.Collections;
using UnityEngine;

public class HandProxyTrialTest : MonoBehaviour
{
    [Header("References")]
    public HandProxyController handController;
    public StraightLineApproachController approachController;
    public GraspOutcomeEvaluator outcomeEvaluator;
    public Transform wristRoot;

    [Header("Sequence")]
    public float pregraspAperture = 0.8f;
    public float wristRollDegrees = 0f;
    public float closeDelay = 0.3f;
    public float liftHeight = 0.12f;
    public float liftSpeed = 0.10f;
    public float holdDuration = 1.0f;

    [Header("Debug")]
    public bool runOnStart = true;

    private IEnumerator Start()
    {
        if (!runOnStart)
        {
            yield break;
        }

        yield return RunTestTrial();
    }

    public IEnumerator RunTestTrial()
    {
        if (handController == null ||
            approachController == null ||
            outcomeEvaluator == null ||
            wristRoot == null)
        {
            Debug.LogError("HandProxyTrialTest references are missing.");
            yield break;
        }

        if (approachController.targetObject == null)
        {
            Debug.LogError("No target object assigned on StraightLineApproachController.");
            yield break;
        }

        outcomeEvaluator.BeginTrial(approachController.targetObject);

        handController.ResetHand();
        handController.SetAperture(pregraspAperture);
        handController.SetWristRoll(wristRollDegrees);

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

        Debug.Log(
            $"Grasp outcome: success={outcome.success}, " +
            $"score={outcome.successScore}, " +
            $"liftHeight={outcome.liftHeight:F3}, " +
            $"heldDistance={outcome.heldDistance:F3}, " +
            $"reason={outcome.reason}"
        );

        Debug.Log("Hand proxy test trial complete.");
    }
}