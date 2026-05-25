# Crack Detection Training

This folder contains crack-detection model training and dataset-preparation scripts.

## Files

### train_ImmerseNet0_32x32.py

Trains the lightweight `ImmerseNet0` model using 32×32 grayscale crack images.

Characteristics:

- Smaller network (2 convolution layers)
- Faster training and inference
- Easier deployment and debugging
- Exports trained weights to C# for AR integration

Expected dataset:

data/cracks_32x32/
├── Positive/
└── Negative/

### train_ImmerseNet1_227x227.py

Trains the larger `ImmerseNet1` model using 227×227 grayscale crack images.

Characteristics:

- Deeper network (4 convolution layers)
- Higher model capacity
- Slower training and inference
- Exports trained weights to C# for AR integration

Expected dataset:

data/original_crack_dataset/
├── Positive/
└── Negative/

### resize_images_to_32x32.py

Utility script that resizes crack images from the original dataset to 32×32 images for use with `train_ImmerseNet0_32x32.py`.

## Dataset

The crack dataset is not included in this repository because of its size.

## Usage

Generate 32×32 images:

python resize_images_to_32x32.py

Train ImmerseNet0:

python train_ImmerseNet0_32x32.py

Train ImmerseNet1:

python train_ImmerseNet1_227x227.py

## Outputs

Training outputs are written to:

outputs/

Including:

- model weights (.pth)
- exported C# weights (.cs)
- hyperparameter reports
- model statistics
