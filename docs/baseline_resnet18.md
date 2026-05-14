# Baseline Model: ResNet-18 (ImageNet Pretrained)

Este documento detalla la arquitectura y las características de `resnet18`, tal y como se carga desde `torchvision.models.resnet18` en este proyecto.

> [!CAUTION]
> **Advertencia Crítica sobre Resolución (CIFAR-10 vs ImageNet)**
> El modelo original de ResNet-18 de PyTorch está diseñado específicamente para imágenes grandes de **ImageNet** (resolución 224x224). 
> Si utilizas este modelo directamente sobre las imágenes pequeñas de **CIFAR-10** (32x32), la primera capa (`conv1` con un kernel de 7x7 y *stride* de 2) y el subsiguiente `maxpool` (kernel de 3x3 y *stride* de 2) reducen agresivamente la resolución espacial de 32x32 a **8x8 píxeles** antes de que empiece el procesamiento real de los bloques residuales.
> Esto causa una pérdida masiva de información espacial y un desplome del *accuracy* inicial. Se recomienda encarecidamente utilizar el modelo `vgg11_bn` (que está adaptado nativamente a CIFAR-10) o sustituir este código por una variante específica tipo `cifar10_resnet18` para obtener resultados válidos.

## 1. Overview
- **Architecture**: ResNet-18 (Residual Network con 18 capas).
- **Dataset de Preentrenamiento**: ImageNet1K (1000 clases).
- **Fuente**: `torchvision.models`.
- **Adaptación Local**: La última capa (`fc`) se reemplaza dinámicamente al construir el modelo para ajustar las salidas al número de clases especificado (por defecto 10 para CIFAR-10). Los pesos de esta nueva capa se inicializan aleatoriamente.
- **Total Parameters**: ~11.1 millones (11,186,570 para 10 clases).

## 2. Layer Breakdown

A diferencia de VGG, ResNet utiliza *BasicBlocks* que contienen conexiones residuales (skip connections). La nomenclatura de rutas de PyTorch es ligeramente diferente.

### 2.1. Feature Extractor (Bloques Convolucionales)
Estos son los objetivos para descomposiciones tensoriales (Tucker, CP, TT, SVD). Las capas de convolución internas se encuentran dentro de las secuencias `layer1`, `layer2`, `layer3` y `layer4`.

**Capa Inicial:**
- `conv1`: **Conv2d(3 -> 64, 7x7, stride=2)** + BatchNorm (`bn1`) + ReLU + MaxPool

**Bloques Residuales:**
Cada `layerX` (del 1 al 4) contiene dos *BasicBlocks* (0 y 1). Cada *BasicBlock* contiene dos convoluciones de 3x3 (`conv1` y `conv2`).

Ejemplos de rutas para apuntar en `config.json`:
- `layer1.0.conv1`: **Conv2d(64 -> 64, 3x3)** (Primer bloque)
- `layer1.0.conv2`: **Conv2d(64 -> 64, 3x3)**
- `layer1.1.conv1`: **Conv2d(64 -> 64, 3x3)**
- `layer1.1.conv2`: **Conv2d(64 -> 64, 3x3)**

- `layer2.0.conv1`: **Conv2d(64 -> 128, 3x3, stride=2)** (Segundo bloque)
- `layer2.0.conv2`: **Conv2d(128 -> 128, 3x3)**
- `layer2.0.downsample.0`: **Conv2d(64 -> 128, 1x1, stride=2)** (Ajuste de dimensiones en la conexión residual)

- `layer3.0.conv1`: **Conv2d(128 -> 256, 3x3, stride=2)**
- `layer4.0.conv1`: **Conv2d(256 -> 512, 3x3, stride=2)**
- `layer4.1.conv2`: **Conv2d(512 -> 512, 3x3)** (Última convolución del extractor)

### 2.2. Classifier (`fc`)
Tras un `AdaptiveAvgPool2d(1x1)` que aplasta la información espacial, se pasa por la capa lineal final. Las descomposiciones aquí se reducen a Truncated SVD.

- `fc`: **Linear(512 -> 10)**

## 3. How to target layers in experiments

En tu archivo de configuración `config.json`, debes utilizar la notación con puntos (dot-notation) exacta para acceder al interior de los bloques residuales.

**Ejemplos válidos:**
- `["conv1"]`: Comprime solo la macro-convolución de entrada (7x7).
- `["layer3.0.conv1", "layer3.0.conv2"]`: Comprime el primer BasicBlock de la capa 3.
- `["fc"]`: Aplica descomposición a la capa lineal del clasificador.
