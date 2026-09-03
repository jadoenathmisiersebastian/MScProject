using UnityEngine;

public class ObjectScaleRandomizer : MonoBehaviour
{
    [Header("Uniform Scale Range")]
    public Vector2 scaleRange = new Vector2(0.8f, 1.2f);

    public void RandomizeScale(GameObject obj)
    {
        if (obj == null)
        {
            return;
        }

        float scale = Random.Range(scaleRange.x, scaleRange.y);
        obj.transform.localScale = obj.transform.localScale * scale;
    }
}