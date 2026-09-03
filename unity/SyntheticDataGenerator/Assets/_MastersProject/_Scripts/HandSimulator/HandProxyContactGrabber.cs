using UnityEngine;

public class HandProxyContactGrabber : MonoBehaviour
{
    [Header("References")]
    public Transform attachPoint;
    public Transform grabZone;

    [Header("Grab State")]
    public bool isClosed;
    public Rigidbody grabbedRigidbody;

    private Rigidbody candidateRigidbody;

    private void OnCollisionEnter(Collision collision)
    {
        Rigidbody rb = collision.rigidbody;

        if (rb == null)
        {
            return;
        }

        if (grabbedRigidbody == null)
        {
            candidateRigidbody = rb;
        }
    }

    public void SetClosed(bool closed)
    {
        isClosed = closed;

        if (isClosed)
        {
            TryGrabCandidate();
        }
        else
        {
            Release();
        }
    }

    private void TryGrabCandidate()
    {
        if (candidateRigidbody == null || grabbedRigidbody != null)
        {
            return;
        }

        if (!IsCandidateInsideGrabZone(candidateRigidbody.gameObject))
        {
            Debug.Log("Candidate touched hand but was outside GrabZone.");
            return;
        }

        grabbedRigidbody = candidateRigidbody;
        grabbedRigidbody.isKinematic = true;

        Transform targetParent = attachPoint != null ? attachPoint : transform;
        grabbedRigidbody.transform.SetParent(targetParent, true);
    }

    private bool IsCandidateInsideGrabZone(GameObject candidate)
    {
        if (grabZone == null)
        {
            Debug.LogWarning("GrabZone is not assigned. Accepting grab by default.");
            return true;
        }

        Bounds candidateBounds = GetObjectBounds(candidate);

        Vector3[] corners =
        {
            new Vector3(candidateBounds.min.x, candidateBounds.min.y, candidateBounds.min.z),
            new Vector3(candidateBounds.min.x, candidateBounds.min.y, candidateBounds.max.z),
            new Vector3(candidateBounds.min.x, candidateBounds.max.y, candidateBounds.min.z),
            new Vector3(candidateBounds.min.x, candidateBounds.max.y, candidateBounds.max.z),
            new Vector3(candidateBounds.max.x, candidateBounds.min.y, candidateBounds.min.z),
            new Vector3(candidateBounds.max.x, candidateBounds.min.y, candidateBounds.max.z),
            new Vector3(candidateBounds.max.x, candidateBounds.max.y, candidateBounds.min.z),
            new Vector3(candidateBounds.max.x, candidateBounds.max.y, candidateBounds.max.z)
        };

        foreach (Vector3 corner in corners)
        {
            Vector3 localCorner = grabZone.InverseTransformPoint(corner);

            if (Mathf.Abs(localCorner.x) <= 0.5f &&
                Mathf.Abs(localCorner.y) <= 0.5f &&
                Mathf.Abs(localCorner.z) <= 0.5f)
            {
                return true;
            }
        }

        Vector3 localCenter = grabZone.InverseTransformPoint(candidateBounds.center);

        return Mathf.Abs(localCenter.x) <= 0.5f &&
               Mathf.Abs(localCenter.y) <= 0.5f &&
               Mathf.Abs(localCenter.z) <= 0.5f;
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

    public void Release()
    {
        if (grabbedRigidbody == null)
        {
            return;
        }

        grabbedRigidbody.transform.SetParent(null, true);
        grabbedRigidbody.isKinematic = false;
        grabbedRigidbody = null;
        candidateRigidbody = null;
    }

    public bool HasObject()
    {
        return grabbedRigidbody != null;
    }
}