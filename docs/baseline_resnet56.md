# Baseline Model: ResNet-56 (CIFAR-10)

This document details the architecture and baseline characteristics of the `resnet56` model used for the tensor decomposition experiments.

## 1. Overview
- **Architecture**: ResNet-56 (Residual Network with 56 layers, designed specifically for CIFAR-10).
- **Dataset**: CIFAR-10 (10 classes, 32x32 RGB images).
- **Total Parameters**: ~850,000 (0.85M)
- **Objective**: This model provides a deeper, highly accurate residual network baseline. It is used to study how tensor decompositions behave on networks with a large number of relatively small-channel residual blocks.

## 2. Architecture Breakdown
Like ResNet-20, ResNet-56 is tailored for the 32x32 resolution of CIFAR-10, avoiding the aggressive early downsampling of ImageNet variants.

The network is composed of:
1. **Initial Convolution**: `Conv2d(3 -> 16, 3x3)`
2. **Stage 1 (32x32 resolution)**: 9 BasicBlocks with 16 channels.
3. **Stage 2 (16x16 resolution)**: 9 BasicBlocks with 32 channels (the first block downsamples with stride=2).
4. **Stage 3 (8x8 resolution)**: 9 BasicBlocks with 64 channels (the first block downsamples with stride=2).
5. **Classifier**: Global Average Pooling followed by a `Linear(64 -> 10)` layer.

### Target Layers
The target layers for tensor decomposition (Tucker, CP, TT, SVD) are primarily the 3x3 convolutions within the `BasicBlocks` of the three stages, as well as the final linear classifier layer. The deeper structure of this model allows for exploring the impact of decomposition depth.
