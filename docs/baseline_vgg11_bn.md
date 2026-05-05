# Baseline Model: VGG11-BatchNorm (CIFAR-10)

This document details the architecture and baseline characteristics of the `vgg11_bn` model used for the tensor decomposition experiments.

## 1. Overview
- **Architecture**: VGG11 with Batch Normalization.
- **Dataset**: CIFAR-10 (10 classes, 32x32 RGB images).
- **Source**: `chenyaofo/pytorch-cifar-models` via PyTorch Hub.
- **Baseline Accuracy**: ~92.0%
- **Total Parameters**: 9,756,426
- **Objective**: Use this pretrained network as a stable, high-accuracy starting point to measure exactly how much parameter reduction and accuracy degradation occurs when applying tensor decompositions (Tucker, CP, TT) to specific layers.

## 2. Layer Breakdown

The model is divided into two main sequential blocks: `features` (Convolutional feature extractor) and `classifier` (Fully Connected layers).

### 2.1. Feature Extractor (`features`)
These are the target layers for 4D Tensor Decompositions (Tucker-2, CP, TT). Note that the tensor weights for each `Conv2d` are 4D tensors of shape `(out_channels, in_channels, kernel_h, kernel_w)`.

- `features.0`: **Conv2d(3 -> 64, 3x3)** + BatchNorm + ReLU + MaxPool
- `features.4`: **Conv2d(64 -> 128, 3x3)** + BatchNorm + ReLU + MaxPool
- `features.8`: **Conv2d(128 -> 256, 3x3)** + BatchNorm + ReLU
- `features.11`: **Conv2d(256 -> 256, 3x3)** + BatchNorm + ReLU + MaxPool
- `features.15`: **Conv2d(256 -> 512, 3x3)** + BatchNorm + ReLU
- `features.18`: **Conv2d(512 -> 512, 3x3)** + BatchNorm + ReLU + MaxPool
- `features.22`: **Conv2d(512 -> 512, 3x3)** + BatchNorm + ReLU
- `features.25`: **Conv2d(512 -> 512, 3x3)** + BatchNorm + ReLU + MaxPool

*After `features.28` (the last MaxPool), the spatial dimensions of the 32x32 CIFAR-10 image have been reduced to 1x1. The tensor shape is `(batch_size, 512, 1, 1)`.*

### 2.2. Classifier (`classifier`)
These layers operate on 2D matrices (Weight shape: `out_features x in_features`). Tensor decompositions applied here mathematically collapse to Truncated SVD, as demonstrated in the previous `MLP` project.

- `classifier.0`: **Linear(512 -> 512)** + ReLU + Dropout
- `classifier.3`: **Linear(512 -> 512)** + ReLU + Dropout
- `classifier.6`: **Linear(512 -> 10)**

## 3. How to target layers in experiments

In the `config.json` file, you reference these layers by their full dot-notation path. 

**Example targets:**
- `["features.0"]`: Compresses only the first convolutional layer (from 3 RGB channels to 64 features).
- `["features.15", "features.18"]`: Compresses intermediate deep convolutional layers.
- `["classifier.0"]`: Applies Truncated SVD to the first dense layer.
