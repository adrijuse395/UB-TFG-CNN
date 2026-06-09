# Baseline Model: ResNet-20 (CIFAR-10)

This document details the architecture and baseline characteristics of the `resnet20` model used for the tensor decomposition experiments.

## 1. Overview
- **Architecture**: ResNet-20 (Residual Network with 20 layers, designed specifically for CIFAR-10).
- **Dataset**: CIFAR-10 (10 classes, 32x32 RGB images).
- **Total Parameters**: ~270,000 (0.27M)
- **Objective**: This model provides a lightweight, modern residual architecture baseline to test how tensor decompositions affect models that are already highly parameter-efficient compared to VGG.

## 2. Architecture Breakdown
Unlike the ImageNet variants of ResNet (which start with a 7x7 convolution and heavy downsampling), the CIFAR-10 ResNets start with a simple 3x3 convolution and maintain the spatial resolution for the first group of residual blocks.

The network is composed of:
1. **Initial Convolution**: `Conv2d(3 -> 16, 3x3)`
2. **Stage 1 (32x32 resolution)**: 3 BasicBlocks with 16 channels.
3. **Stage 2 (16x16 resolution)**: 3 BasicBlocks with 32 channels (the first block downsamples with stride=2).
4. **Stage 3 (8x8 resolution)**: 3 BasicBlocks with 64 channels (the first block downsamples with stride=2).
5. **Classifier**: Global Average Pooling followed by a `Linear(64 -> 10)` layer.

### Target Layers
The target layers for tensor decomposition (Tucker, CP, TT, SVD) are primarily the 3x3 convolutions within the `BasicBlocks` of the three stages, as well as the final linear classifier layer.
