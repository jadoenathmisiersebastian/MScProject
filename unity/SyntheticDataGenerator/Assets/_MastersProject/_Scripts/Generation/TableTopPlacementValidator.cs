// using UnityEngine;

// public class TabletopPlacementValidator : MonoBehaviour
// {
//     public Transform captureBoundsTransform;

//     public bool IsObjectInsideCaptureBounds(GameObject obj)
//     {
//         if (captureBoundsTransform == null)
//         {
//             Debug.LogError("Capture bounds transform is not assigned.");
//             return false;
//         }

//         Bounds captureBounds = new Bounds(
//             captureBoundsTransform.position,
//             captureBoundsTransform.lossyScale
//         );

//         Renderer[] renderers = obj.GetComponentsInChildren<Renderer>();

//         if (renderers.Length == 0)
//         {
//             Debug.LogWarning($"Object {obj.name} has no renderers.");
//             return false;
//         }

//         Bounds objectBounds = renderers[0].bounds;

//         for (int i = 1; i < renderers.Length; i++)
//         {
//             objectBounds.Encapsulate(renderers[i].bounds);
//         }

//         return captureBounds.Contains(objectBounds.min) &&
//                captureBounds.Contains(objectBounds.max);
//     }


//     public bool IsObjectSettled(GameObject obj, float maxVelocity = 0.02f, float maxAngularVelocity = 0.05f)
//     {
//         Rigidbody[] rigidbodies = obj.GetComponentsInChildren<Rigidbody>();

//         if (rigidbodies.Length == 0)
//         {
//             return true;
//         }

//         foreach (Rigidbody rb in rigidbodies)
//         {
//             if (rb.linearVelocity.magnitude > maxVelocity)
//             {
//                 return false;
//             }

//             if (rb.angularVelocity.magnitude > maxAngularVelocity)
//             {
//                 return false;
//             }
//         }

//         return true;
//     }

// }


using UnityEngine;

public class TabletopPlacementValidator : MonoBehaviour
{
    public Transform captureBoundsTransform;

    public bool IsObjectInsideCaptureBounds(GameObject obj)
    {
        return GetPlacementInvalidReason(obj) == null;
    }

    public string GetPlacementInvalidReason(GameObject obj)
    {
        if (obj == null)
        {
            return "Object is null.";
        }

        if (captureBoundsTransform == null)
        {
            return "CaptureBounds transform is not assigned.";
        }

        Renderer[] renderers = obj.GetComponentsInChildren<Renderer>();

        if (renderers.Length == 0)
        {
            return $"Object {obj.name} has no renderers.";
        }

        Bounds objectBounds = renderers[0].bounds;

        for (int i = 1; i < renderers.Length; i++)
        {
            objectBounds.Encapsulate(renderers[i].bounds);
        }

        Bounds captureBounds = new Bounds(
            captureBoundsTransform.position,
            captureBoundsTransform.lossyScale
        );

        if (!captureBounds.Contains(objectBounds.min) ||
            !captureBounds.Contains(objectBounds.max))
        {
            return
                $"Object bounds outside CaptureBounds. " +
                $"Object min={objectBounds.min}, max={objectBounds.max}. " +
                $"Capture min={captureBounds.min}, max={captureBounds.max}.";
        }

        return null;
    }

    public bool IsObjectSettled(
        GameObject obj,
        float maxVelocity = 0.005f,
        float maxAngularVelocity = 0.02f
    )
    {
        Rigidbody[] rigidbodies = obj.GetComponentsInChildren<Rigidbody>();

        if (rigidbodies.Length == 0)
        {
            return true;
        }

        foreach (Rigidbody rb in rigidbodies)
        {
            if (rb.IsSleeping())
            {
                continue;
            }

            if (rb.linearVelocity.magnitude > maxVelocity)
            {
                return false;
            }

            if (rb.angularVelocity.magnitude > maxAngularVelocity)
            {
                return false;
            }
        }

        return true;
    }
}