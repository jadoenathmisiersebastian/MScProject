using System;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

[RequireComponent(typeof(Light))] //Can only attach to GameObjects which also have a Light component attached
//This tag is used to "target" which objects in the scene will be randomized
public class LightRandomizerTag : RandomizerTag { }

[Serializable]
[AddRandomizerMenu("LightRandomizer")]
public class LightRandomizer : Randomizer
{

    public Transform lookAtTarget;

    public Vector3Parameter lightPosition = new Vector3Parameter
    {
        x = new UniformSampler(-5f, 5f),
        y = new UniformSampler(3f, 8f),   // height above table
        z = new UniformSampler(-5f, 5f)
    };

    public ColorRgbParameter color;

    // A parameter whose value uniformly ranges from 2 to 10 when sampled
    public FloatParameter lightIntensity = new() { value = new UniformSampler(0, 1) };

    // Run this every randomization iteration
    protected override void OnIterationStart()
    {
        // Get all MyLightRandomizerTag's in the scene
        var tags = tagManager.Query<LightRandomizerTag>();
        foreach (var tag in tags)
        {
            // Get the light attached to the object
            var tagLight = tag.GetComponent<Light>();
            tagLight.intensity = lightIntensity.Sample();
            tagLight.color = color.Sample();

            tag.transform.position = lightPosition.Sample();

            tag.transform.LookAt(lookAtTarget);

        }
    }
}