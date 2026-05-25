# Crack Segmentation Training

This folder contains a U-Net based crack segmentation pipeline.

Run all commands from inside Training/Crack_Segmentation so the relative paths work correctly.

## Files

main.py: trains the U-Net model using paired crack images and masks.

model.py: defines the U-Net segmentation model.

transform_into_C#Unity.py: final step that converts the trained U-Net checkpoint into C# weight arrays compatible with the C#-Unity platform for AR headset deployment.

dataset.py, metrics.py, evaluation.py: support files for dataset loading, metrics, and evaluation.

## Dataset

The dataset is not included.

Expected structure:

dataset/train_images/
dataset/train_masks/
dataset/test_images/
dataset/test_masks/

Image and mask filenames must match by stem.

## Train

python main.py --image-dir dataset/train_images --mask-dir dataset/train_masks --save-dir checkpoints --device cpu

## Export C# weights

python transform_into_C#Unity.py --checkpoint-path checkpoints/fold_1/best_model.pth --output-dir csharp_export
