# LeNetAR: Deep Learning Training and Deployment on AR Headsets

This repository contains deep learning models, training pipelines, and deployment examples developed for Microsoft HoloLens augmented reality headsets.

The repository demonstrates the complete workflow from model training in Python to deployment in Unity using C#.

## Overview

The project was originally developed to deploy a LeNet-5 model trained on the MNIST dataset to AR headsets. The repository has since been expanded to include crack classification and crack segmentation models following the same deployment workflow.

The general process is:

1. Train and evaluate the model in Python.
2. Export trained weights to a C#-compatible format.
3. Integrate the exported weights into a Unity project.
4. Deploy the application to Microsoft HoloLens using UWP.

## Repository Structure

### Training

Python training, evaluation, and weight-export files are maintained in a separate repository:

**Python-Training-for-Unity-AR**

https://github.com/kmalek-eng/Python-Training-for-Unity-AR

This repository contains the training pipelines for:

* LeNet-5 MNIST classification
* Crack classification
* Crack segmentation

### LeNetAR

Unity implementation of the LeNet-5 MNIST classifier for Microsoft HoloLens and HoloLens 2.

The project is developed using:

* Unity
* C#
* UnityEngine
* Universal Windows Platform (UWP)

The same deployment approach can be applied to the crack classification and crack segmentation models after exporting their trained weights to C# format.

## Requirements

### Deployment

* Unity 2019.4.40f1
* Microsoft HoloLens or HoloLens 2
* Windows UWP SDK

## Setup

Clone the repository:

```bash
git clone https://github.com/kmalek-eng/LeNetAR.git
cd LeNetAR
```

To open the AR application:

1. Launch Unity Hub.
2. Add the LeNetAR project.
3. Open the project in Unity.
4. Set the build target to UWP.
5. Build and deploy to HoloLens.

## Usage

Once deployed, the application will:

* Recognize handwritten digits through the AR headset.
* Display the recognition results in real-time within the AR environment.

## Citation

If you use this code, please cite the author:

Kaveh Malek

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* Unity Technologies
* Microsoft HoloLens
* MNIST Dataset
* Amr Kassaem for enabling built-in voice recognition commands




https://github.com/KaMa85/LeNet/assets/82784239/7ebb861a-7852-4702-925f-e89c9f4d81e4



