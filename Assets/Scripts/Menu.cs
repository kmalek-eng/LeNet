using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit.UI;
public class Menu : MonoBehaviour
{
    //Add scripts that interact with the menu
    public LeNet_Kaveh CM;

    public int [] sliderMultipliers;
    //All input items of the menu
    public PinchSlider[] sliders;
    public Interactable[] buttons;

    private void GiveStringToSlider(int index, string th)
    {
        PinchSlider slider = sliders[index];
        string value = th;
        slider.transform.parent.Find("Value").GetComponent<TextMesh>().text = value;
    }
    private void GiveIntToSlider(int index,  int th)
    {
        PinchSlider slider = sliders[index];
        int value = th;
        slider.transform.parent.Find("Value").GetComponent<TextMesh>().text = value.ToString("#.##");
    }
    private void GiveFloatToSlider(int index,float  th)
    {
        PinchSlider slider = sliders[index];
        float value = th;
        slider.transform.parent.Find("Value").GetComponent<TextMesh>().text = value.ToString("#.##");
    }

    private void GiveDoubleToSlider(int index, double th)
    {
        PinchSlider slider = sliders[index];
        double value = th;
        slider.transform.parent.Find("Value").GetComponent<TextMesh>().text = value.ToString("#.##");
    }

    //private int GetIntFromSlider(int index)
    //{
    //    PinchSlider slider = sliders[index];
    //    int value = (int)(slider.SliderValue * sliderMultipliers[index-3]);
    //    slider.transform.parent.Find("Value").GetComponent<TextMesh>().text = sliderMultipliers[index - 3].ToString();
    //    return value;
    //}
    //public void UpdateDetectLineCannyValue()
    //{
    //    CM.val = GetIntFromSlider(3);
    //}

    // Start is called before the first frame update
    void Start()
    {

    }
    // Update is called once per frame
    void Update()
    {
        GiveDoubleToSlider(0, CM.pred);
    }
}
