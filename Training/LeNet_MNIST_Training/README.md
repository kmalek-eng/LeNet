# LeNet MNIST Training

This folder contains the LeNet-5 training script for MNIST digit classification.

## Files

### train_LeNet5_MNIST.py

Trains a LeNet-5 model using MNIST images padded from 28x28 to 32x32.

The script:

- downloads/loads MNIST using `torchvision.datasets.MNIST`
- trains LeNet-5
- tests the trained model
- generates a confusion matrix and ROC curves
- saves PyTorch weights
- exports model weights to Excel
- exports C# weight arrays for AR/Unity use
- exports sample MNIST images as C# arrays

## Dataset

Expected dataset folder:

data/MNIST/raw

MNIST is loaded through:

datasets.MNIST(root="./datasets/", train=True, download=True)

So the active runtime dataset path is:

datasets/MNIST/

## Outputs

Training outputs are saved in:

outputs/
