using UnityEngine;

public class HandProxyController : MonoBehaviour
{
    [Header("References")]
    public Transform wristRoot;
    public Transform fingerLeft;
    public Transform fingerRight;

    [Header("Aperture")]
    public float minAperture = 0.025f;
    public float maxAperture = 0.09f;
    [Range(0f, 1f)]
    public float currentAperture = 1.0f;

    [Header("Wrist")]
    public float currentWristRollDegrees = 0f;


    public HandProxyContactGrabber contactGrabber;

    private Vector3 initialWristLocalPosition;
    private Quaternion initialWristLocalRotation;

    private void Awake()
    {
        if (wristRoot != null)
        {
            initialWristLocalPosition = wristRoot.localPosition;
            initialWristLocalRotation = wristRoot.localRotation;
        }

        Open();
    }

    public void Open()
    {
        SetAperture(1.0f);

        if (contactGrabber != null)
        {
            contactGrabber.SetClosed(false);
        }
    }

    public void Close()
    {
        SetAperture(0.0f);

        if (contactGrabber != null)
        {
            contactGrabber.SetClosed(true);
        }
    }

    private void Update()
    {
        SetAperture(currentAperture);
        SetWristRoll(currentWristRollDegrees);
    }

    public void SetAperture(float aperture01)
    {
        currentAperture = Mathf.Clamp01(aperture01);

        float halfDistance = Mathf.Lerp(
            minAperture * 0.5f,
            maxAperture * 0.5f,
            currentAperture
        );

        if (fingerLeft != null)
        {
            Vector3 localPosition = fingerLeft.localPosition;
            localPosition.x = -halfDistance;
            fingerLeft.localPosition = localPosition;
        }

        if (fingerRight != null)
        {
            Vector3 localPosition = fingerRight.localPosition;
            localPosition.x = halfDistance;
            fingerRight.localPosition = localPosition;
        }
    }

    public void SetWristRoll(float rollDegrees)
    {
        currentWristRollDegrees = rollDegrees;

        if (wristRoot == null)
        {
            return;
        }

        wristRoot.localRotation =
            initialWristLocalRotation *
            Quaternion.AngleAxis(currentWristRollDegrees, Vector3.forward);
    }

    public void ResetHand()
    {
        if (wristRoot != null)
        {
            wristRoot.localPosition = initialWristLocalPosition;
            wristRoot.localRotation = initialWristLocalRotation;
        }

        currentWristRollDegrees = 0f;
        Open();
    }
}
