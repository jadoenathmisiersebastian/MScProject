//using System;
//using UnityEngine;
//using UnityEngine.Perception.Randomization.Parameters;
//using UnityEngine.Perception.Randomization.Randomizers;
//using UnityEngine.Perception.Randomization.Samplers;

//public class CameraRandomizerTag : RandomizerTag { }

//[Serializable]
//[AddRandomizerMenu("Custom/Camera Randomizer")]
//public class CameraRandomizer : Randomizer
//{
//    public Transform target; // table centre

//    public Vector3Parameter cameraPosition = new Vector3Parameter
//    {
//        x = new UniformSampler(-3f, 3f),
//        y = new UniformSampler(2f, 5f),
//        z = new UniformSampler(-3f, 3f)
//    };

//    protected override void OnIterationStart()
//    {
//        var tags = tagManager.Query<CameraRandomizerTag>();

//        foreach (var tag in tags)
//        {
//            Vector3 pos = cameraPosition.Sample();

//            tag.transform.position = pos;

//            // 🔥 Always look at table centre
//            tag.transform.LookAt(target);
//        }
//    }
//}




using System;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

public class CameraRandomizerTag : RandomizerTag { }

[Serializable]
[AddRandomizerMenu("Custom/Camera Randomizer")]
public class CameraRandomizer : Randomizer
{
    public Transform target; // table centre

    public Vector3Parameter cameraPosition = new Vector3Parameter
    {
        x = new UniformSampler(-3f, 3f),
        y = new UniformSampler(2f, 5f),
        z = new UniformSampler(-3f, 3f)
    };

    protected override void OnIterationStart()
    {
        var tags = tagManager.Query<CameraRandomizerTag>();

        foreach (var tag in tags)
        {
            Vector3 pos = cameraPosition.Sample();

            tag.transform.position = pos;

            // 🔥 Always look at table centre
            Vector3 jitter = new Vector3(
                UnityEngine.Random.Range(-0.2f, 0.2f),
                UnityEngine.Random.Range(-0.2f, 0.2f),
                UnityEngine.Random.Range(-0.2f, 0.2f)
            );

            tag.transform.LookAt(target.position + jitter);
        }
    }
}