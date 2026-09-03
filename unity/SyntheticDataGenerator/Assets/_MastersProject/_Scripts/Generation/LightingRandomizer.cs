using UnityEngine;

public class LightingRandomizer : MonoBehaviour
{
    [Header("Lights")]
    public Light keyLight;
    public Light fillLight;
    public Light overheadLight;

    [Header("Key Light Intensity")]
    public Vector2 keyIntensityRange = new Vector2(700f, 1400f);

    [Header("Fill Light Intensity")]
    public Vector2 fillIntensityRange = new Vector2(100f, 500f);

    [Header("Overhead Light Intensity")]
    public Vector2 overheadIntensityRange = new Vector2(100f, 600f);

    [Header("Color Temperature")]
    public Vector2 temperatureRange = new Vector2(4500f, 7000f);

    public void RandomizeLighting()
    {
        RandomizeLight(keyLight, keyIntensityRange);
        RandomizeLight(fillLight, fillIntensityRange);
        RandomizeLight(overheadLight, overheadIntensityRange);
    }

    private void RandomizeLight(Light targetLight, Vector2 intensityRange)
    {
        if (targetLight == null)
        {
            return;
        }

        targetLight.intensity = Random.Range(
            intensityRange.x,
            intensityRange.y
        );

        targetLight.useColorTemperature = true;
        targetLight.colorTemperature = Random.Range(
            temperatureRange.x,
            temperatureRange.y
        );
    }
}