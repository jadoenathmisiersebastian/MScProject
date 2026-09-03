using UnityEngine;

public class ObjectKillPlane : MonoBehaviour
{
    public bool WasTriggered { get; private set; }

    public void ResetState()
    {
        WasTriggered = false;
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.attachedRigidbody != null)
        {
            WasTriggered = true;
        }
    }
}