using System;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

[RequireComponent(typeof(Light))]
public class LightRandomizerTag : RandomizerTag { }

[Serializable]
[AddRandomizerMenu("LightRandomizer")]
public class LightRandomizer : Randomizer
{

    public Transform lookAtTarget;

    public Vector3Parameter lightPosition = new Vector3Parameter
    {
        x = new UniformSampler(-5f, 5f),
        y = new UniformSampler(3f, 8f),
        z = new UniformSampler(-5f, 5f)
    };

    public ColorRgbParameter color;

    public FloatParameter lightIntensity = new() { value = new UniformSampler(0, 1) };

    protected override void OnIterationStart()
    {
        var tags = tagManager.Query<LightRandomizerTag>();
        foreach (var tag in tags)
        {
            var tagLight = tag.GetComponent<Light>();
            tagLight.intensity = lightIntensity.Sample();
            tagLight.color = color.Sample();

            tag.transform.position = lightPosition.Sample();

            tag.transform.LookAt(lookAtTarget);

        }
    }
}
