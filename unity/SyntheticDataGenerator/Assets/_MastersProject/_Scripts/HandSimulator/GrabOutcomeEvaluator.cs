using UnityEngine;

public class GraspOutcomeEvaluator : MonoBehaviour
{
    [Header("References")]
    public HandProxyContactGrabber contactGrabber;
    public Transform palmTransform;
    public Transform targetObject;

    [Header("Success Thresholds")]
    public float requiredLiftHeight = 0.08f;
    public float holdDistanceThreshold = 0.18f;

    private float initialObjectY;

    public void BeginTrial(Transform target)
    {
        targetObject = target;

        if (targetObject != null)
        {
            initialObjectY = GetObjectCenter(targetObject).y;
        }
    }

    public GraspOutcome Evaluate()
    {
        if (targetObject == null)
        {
            return new GraspOutcome
            {
                success = false,
                successScore = 0f,
                liftHeight = 0f,
                heldDistance = float.PositiveInfinity,
                reason = "No target object assigned."
            };
        }

        Vector3 objectCenter = GetObjectCenter(targetObject);
        Vector3 palmCenter = palmTransform != null ? palmTransform.position : Vector3.zero;

        float liftHeight = objectCenter.y - initialObjectY;
        float heldDistance = Vector3.Distance(objectCenter, palmCenter);

        bool grabbed = contactGrabber != null && contactGrabber.HasObject();
        bool lifted = liftHeight >= requiredLiftHeight;
        bool held = heldDistance <= holdDistanceThreshold;

        bool success = grabbed && lifted && held;

        return new GraspOutcome
        {
            success = success,
            successScore = success ? 1f : 0f,
            liftHeight = Mathf.Max(0f, liftHeight),
            heldDistance = heldDistance,
            grabbed = grabbed,
            lifted = lifted,
            held = held,
            reason = success ? "success" : BuildFailureReason(grabbed, lifted, held)
        };
    }

    private string BuildFailureReason(bool grabbed, bool lifted, bool held)
    {
        if (!grabbed)
        {
            return "object_not_grabbed";
        }

        if (!lifted)
        {
            return "object_not_lifted";
        }

        if (!held)
        {
            return "object_not_held_near_palm";
        }

        return "unknown_failure";
    }

    private Vector3 GetObjectCenter(Transform obj)
    {
        Renderer[] renderers = obj.GetComponentsInChildren<Renderer>();

        if (renderers.Length > 0)
        {
            Bounds bounds = renderers[0].bounds;

            for (int i = 1; i < renderers.Length; i++)
            {
                bounds.Encapsulate(renderers[i].bounds);
            }

            return bounds.center;
        }

        return obj.position;
    }
}

[System.Serializable]
public class GraspOutcome
{
    public bool success;
    public float successScore;
    public float liftHeight;
    public float heldDistance;

    public bool grabbed;
    public bool lifted;
    public bool held;

    public string reason;
}