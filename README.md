# Compression of Convolutional Neural Networks using Tensor Decomposition

> **Author:** Adrià Junquera Selma  
> **Degree:** Computer Engineering (Bachelor's Thesis / TFG - Part 2)  
> **University:** Universitat de Barcelona

## 1. Context and Motivation
This project aims to study the empirical compression of Convolutional Neural Networks (CNN) using tensor decomposition techniques, with a direct application to **TinyML** (devices with highly limited computational and memory resources like microcontrollers, edge devices, and smart sensors).

Modern CNNs are heavily overparameterized. By decomposing the weight tensors of computationally expensive layers into products of lower-rank factors, we can significantly reduce the number of stored parameters and, in some cases, the required number of operations for inference.

This repository explores and compares four different methods:
- **SVD (Singular Value Decomposition)**: Serves as a 2D baseline.
- **Tucker-2 Decomposition**: Applies compression over the two channel modes (input and output) while preserving the spatial dimensions.
- **CP (Canonical Polyadic) Decomposition**: Decomposes the tensor into a sum of rank-1 tensors.
- **TT (Tensor Train)**: Decomposes the tensor into a chain of 3rd-order core tensors.

## 2. Models and Datasets
The primary experiments focus on the **CIFAR-10** and **CIFAR-100** datasets, evaluating networks such as **VGG11-BN**, **ResNet-20**, and **ResNet-56**. 

The target layers for compression are all the convolutional and linear layers of the models (excluding Batch Normalization and MaxPool layers).

> **Note:** For more detailed architectural breakdowns of the baseline models (VGG11-BN, ResNet-20, ResNet-56), please check the `docs/` directory.

## 3. Experimental Pipeline
The evaluation pipeline replaces the target layers in the base model with their compressed counterparts. It supports:
- **Compression**: Decomposes the pre-trained weights to a specified rank.
- **Fine-tuning**: A post-compression re-training phase (using Adam optimizer and a dynamic learning rate schedule) to recover lost accuracy.
- **Evaluation**: Computes metrics like accuracy, total parameters, compression ratio, peak inference memory, MACs, latency, and throughput.

## 4. How to Execute the Program

### Prerequisites
Make sure to install the required dependencies (preferably within a virtual environment):
```bash
pip install -r requirements.txt
```

### Running Experiments
The entry point for the pipeline is `main.py`. The execution requires a configuration JSON file which specifies the dataset, model, decomposition methods, ranks, and fine-tuning settings.

All configuration files are located in the `configs/` directory.

To run an experiment, simply execute:
```bash
python main.py --config configs/<config_file>.json
```

**Examples:**
- Run a quick smoke test on CIFAR-100:
  ```bash
  python main.py --config configs/config_cifar100_smoke.json
  ```
- Run a full ResNet-20 experiment:
  ```bash
  python main.py --config configs/config_resnet20.json
  ```

### Plotting and Results
The experiments will generate output folders under `runs/run_YYYYMMDD_HHMMSS/` containing a `results.csv` and a copy of the input config. 

To generate visual plots comparing the different compression methods, run:
```bash
python run_plots.py runs/<your_run_directory_name>
```
The generated plots will be saved inside the `plots/` folder of the run directory.

## 5. Repository Structure
```
CNN/
├── configs/                  # JSON configuration files for experiments
├── src/
│   ├── decompositions/       # Implementations of SVD, Tucker, CP, and TT
│   ├── training/             # Fine-tuning logic and dynamic LR scheduling
│   ├── evaluation/           # Metrics calculation and ModelEvaluator
│   ├── models/               # Model architectures and factory
│   └── data/                 # Datasets loading and processing
├── scripts/                  # Plotting and analysis scripts
├── experiments/              # Helper scripts for generating configurations
├── runs/                     # Output directory for experiment results
├── main.py                   # Main pipeline execution script
├── run_plots.py              # Main plotting execution script
└── requirements.txt          # Python dependencies
```
