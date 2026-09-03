using System.Collections.Generic;
using UnityEngine;

public class GraspCandidateGenerator : MonoBehaviour
{
    public string graspType = "cylindrical";
    public float handAperture = 0.6f;
    public string handApertureLabel = "medium";
    public float[] wristRollDegrees = { -60f, -30f, 0f, 30f, 60f };

    public List<GraspCandidate> GenerateCandidates()
    {
        List<GraspCandidate> candidates = new List<GraspCandidate>();

        foreach (float roll in wristRollDegrees)
        {
            string rollText = roll >= 0f ? $"+{roll:0}" : $"{roll:0}";
            string candidateId = $"{graspType}_roll{rollText}_{handApertureLabel}";

            candidates.Add(new GraspCandidate(
                candidateId,
                graspType,
                roll,
                handAperture,
                handApertureLabel,
                new Vector3(0f, 0f, 1f)
            ));
        }

        return candidates;
    }
}