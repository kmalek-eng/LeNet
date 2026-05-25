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

Contains Python implementations for model training, evaluation, and weight export.

#### LeNet_MNIST_Training

LeNet-5 implementation for handwritten digit classification using the MNIST dataset.

Includes:

- Training and testing
- Accuracy evaluation
- Confusion matrix and ROC analysis
- Export of trained weights to C#
- Export of sample images for verification in Unity

#### Crack_Detection_Training

Binary crack classification models based on the ImmerseNet architecture.

Includes:

- Dataset preprocessing
- Model training and evaluation
- Weight export to C#
- A lightweight 32×32 implementation for faster development and deployment
- A larger 227×227 implementation for higher-capacity training

#### Crack_Segmentation

U-Net based crack segmentation framework.

Includes:

- Training using image-mask pairs
- Cross-validation
- Model evaluation
- Segmentation result generation
- Checkpoint generation
- Conversion of trained weights into a C#-Unity compatible format for AR deployment

Each training folder contains its own README describing dataset structure, execution steps, and generated outputs.

### LeNetAR

Unity implementation of the LeNet-5 MNIST classifier for Microsoft HoloLens and HoloLens 2.

The project is developed using:

- Unity
- C#
- UnityEngine
- Universal Windows Platform (UWP)

The same deployment approach can be applied to the crack classification and crack segmentation models after exporting their trained weights to C# format.

## Requirements

### Training

- Python 3
- PyTorch
- NumPy
- scikit-learn
- OpenCV
- Pillow

### Deployment

- Unity 2019.4.40f1
- Microsoft HoloLens or HoloLens 2
- Windows UWP SDK

## Setup

Clone the repository:

```bash
git clone https://github.com/KaMa85/LeNet.git
cd LeNet
```

Refer to the README inside each training folder for dataset requirements and execution instructions.

To open the AR application:

1. Launch Unity Hub.
2. Add the LeNetAR project.
3. Open the project in Unity.
4. Set the build target to UWP.
5. Build and deploy to HoloLens.

## Citation

If you use this code, please cite:

Kaveh Malek

## License

This project is licensed under the MIT License.

## Acknowledgments

- Unity Technologies
- Microsoft HoloLens
- MNIST Dataset
- Amr Kassaem for enabling built-in voice recognition commands



https://github.com/KaMa85/LeNet/assets/82784239/7ebb861a-7852-4702-925f-e89c9f4d81e4



