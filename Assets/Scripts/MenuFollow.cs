using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MenuFollow : MonoBehaviour
{
    public float minAngle=25.0f;
    public float rotSpeed =10.0f;
    public float lerpFactor=5.0f;

    // Update is called once per frame
    void Update()
    {
        if (Vector3.Angle(transform.forward, Camera.main.transform.forward) > minAngle)
        {
            transform.rotation = Quaternion.RotateTowards(transform.rotation, Camera.main.transform.rotation, rotSpeed * Time.deltaTime);
        }
        transform.position = Vector3.Lerp(transform.position, Camera.main.transform.position, Time.deltaTime);
    }

    public void Reset()
    {
        transform.position = Camera.main.transform.position;
        transform.rotation = Camera.main.transform.rotation;
    }
}
