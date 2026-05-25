# Crack Detection Training

This folder contains the training code for binary crack detection using ImmerseNet0.

## Dataset

The dataset is not included in this repository because of its size.

Expected structure:

data/cracks_32x32/
├── Positive/
└── Negative/

Place crack images in `Positive` and non-crack images in `Negative`.

## Run

python crack.py

Training outputs are saved in:

outputs/
