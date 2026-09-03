using System;
using UnityEngine;

[Serializable]
public class GraspCandidate
{
    public string candidateId;
    public string graspType;
    public float wristRollDegrees;
    [Range(0f, 1f)]
    public float handAperture;
    public string handApertureLabel;
    public Vector3 approachDirectionCamera;

    public GraspCandidate(
        string candidateId,
        string graspType,
        float wristRollDegrees,
        float handAperture,
        string handApertureLabel,
        Vector3 approachDirectionCamera
    )
    {
        this.candidateId = candidateId;
        this.graspType = graspType;
        this.wristRollDegrees = wristRollDegrees;
        this.handAperture = handAperture;
        this.handApertureLabel = handApertureLabel;
        this.approachDirectionCamera = approachDirectionCamera;
    }
}