using System.Collections;
using UnityEngine;

public class StraightLineApproachController : MonoBehaviour
{
    [Header("References")]
    public Transform wristRoot;
    public Transform targetObject;

    [Header("Approach")]
    public float startDistance = 0.35f;
    public float stopDistance = 0.06f;
    public float approachSpeed = 0.15f;

    public void PlaceAtStartPose()
    {
        if (wristRoot == null || targetObject == null)
        {
            Debug.LogError("StraightLineApproachController references are missing.");
            return;
        }

        Vector3 approachDirection = GetApproachDirection();
        Vector3 targetCenter = GetTargetCenter();

        wristRoot.position = targetCenter - approachDirection * startDistance;
        wristRoot.rotation = Quaternion.LookRotation(approachDirection, Vector3.up);
    }

    public IEnumerator MoveToClosurePose()
    {
        if (wristRoot == null || targetObject == null)
        {
            yield break;
        }

        Vector3 approachDirection = GetApproachDirection();
        Vector3 targetCenter = GetTargetCenter();
        Vector3 closurePosition = targetCenter - approachDirection * stopDistance;

        while (Vector3.Distance(wristRoot.position, closurePosition) > 0.005f)
        {
            wristRoot.position = Vector3.MoveTowards(
                wristRoot.position,
                closurePosition,
                approachSpeed * Time.deltaTime
            );

            wristRoot.rotation = Quaternion.LookRotation(approachDirection, Vector3.up);

            yield return null;
        }
    }

    private Vector3 GetApproachDirection()
    {
        // First version: approach from camera/front side toward positive Z.
        return Vector3.forward;
    }

    private Vector3 GetTargetCenter()
    {
        Renderer renderer = targetObject.GetComponentInChildren<Renderer>();

        if (renderer != null)
        {
            return renderer.bounds.center;
        }

        return targetObject.position;
    }
}