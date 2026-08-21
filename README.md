# CNN Model Zoo with PyTorch

A simple PyTorch implementation of classic convolutional neural network architectures trained on CIFAR-10.

This project implements several well-known CNN architectures from scratch for educational purposes. The models share a common training and evaluation pipeline, making it easy to study their architectures and compare their behavior under the same dataset and training environment.

## Features

- Classic CNN architectures implemented from scratch with PyTorch
- CIFAR-10 image classification
- Unified training and evaluation pipeline
- Model selection through command-line arguments
- Automatic CIFAR-10 download
- CUDA support when an NVIDIA GPU is available
- Reproducible random seed settings
- Data augmentation
- Training and test accuracy calculation
- Training and test loss calculation

## Supported Models

The current version includes the following architectures:

| Argument | Architecture |
|---|---|
| `alexnet` | AlexNet adapted for CIFAR-10 |
| `vgg` | VGG-style convolutional network |
| `googlenet` | GoogLeNet / Inception network |
| `resnet` | ResNet-18-style residual network |

The architectures are implemented directly with PyTorch modules rather than using pretrained models from `torchvision.models`.

Some input layers are modified from the original ImageNet architectures to better support the smaller `32 × 32` CIFAR-10 images.

## Project Structure

```text
cnn-model-zoo-pytorch/
├── datasets/
│   └── cifar.py           # CIFAR-10 loading, preprocessing, and DataLoaders
├── models/
│   ├── alexnet.py         # AlexNet implementation
│   ├── densenet.py        # DenseNet implementation (planned)
│   ├── googlenet.py       # GoogLeNet and Inception implementation
│   ├── resnet.py          # ResNet and residual block implementation
│   └── vgg.py             # VGG implementation
├── utils/
│   ├── seed.py            # Reproducible random seed settings
│   ├── trainer.py         # Training and evaluation functions
│   └── visualize.py       # Visualization utilities (planned)
├── main.py                # Main training script and command-line interface
├── requirements.txt       # Python dependencies
├── README.md
├── LICENSE
└── .gitignore
```

## Dataset

This project currently uses CIFAR-10.

CIFAR-10 contains 60,000 RGB images belonging to 10 classes.

| Split | Number of images |
|---|---:|
| Training | 50,000 |
| Test | 10,000 |
| Total | 60,000 |

Each image has the following shape:

```text
3 × 32 × 32
```

The 10 classes are:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

The dataset is downloaded automatically through `torchvision.datasets.CIFAR10`.

## Data Augmentation

The training pipeline currently applies:

```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.ToTensor()
transforms.Normalize(
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5)
)
transforms.RandomErasing(
    p=0.5,
    scale=(0.02, 0.33),
    ratio=(0.3, 3.3)
)
```

The test dataset is normalized but is not augmented.

## Requirements

- Python 3.10 or later
- PyTorch
- torchvision
- NumPy
- tqdm
- torchinfo
- Git

An NVIDIA GPU is optional.

The program automatically uses CUDA when a CUDA-enabled GPU is available and otherwise runs on the CPU.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Manstein0704/cnn-model-zoo-pytorch.git
cd cnn-model-zoo-pytorch
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

#### macOS or Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch

For CPU execution:

```bash
pip install torch torchvision
```

For NVIDIA GPU execution, install a CUDA-enabled version of PyTorch that matches your environment using the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

### 4. Install the remaining dependencies

```bash
pip install numpy tqdm torchinfo
```

## Verify the GPU Environment

Run:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Example output with an NVIDIA GPU:

```text
PyTorch: 2.x.x+cuXXX
CUDA available: True
GPU: NVIDIA GeForce RTX 4060
```

## Usage

Run the training script from the repository root.

The default model is ResNet:

```bash
python main.py
```

A specific architecture can be selected using `--model`.

### AlexNet

```bash
python main.py --model alexnet
```

### VGG

```bash
python main.py --model vgg
```

### GoogLeNet

```bash
python main.py --model googlenet
```

### ResNet

```bash
python main.py --model resnet
```

To display all available options:

```bash
python main.py --help
```

## Command-Line Arguments

The current training configuration can be controlled through command-line arguments.

| Argument | Description | Default |
|---|---|---:|
| `--model` | Model architecture | `resnet` |
| `--seed` | Random seed | `2026` |
| `--batch_size` | Mini-batch size | `32` |
| `--num_epochs` | Number of training epochs | `500` |
| `--lr` | Learning rate | `0.001` |

For example:

```bash
python main.py --model resnet --num_epochs 100 --batch_size 128 --lr 0.001 --seed 2026
```

For a quick execution test:

```bash
python main.py --model resnet --num_epochs 1
```

## Training Pipeline

The basic training flow is:

```text
CIFAR-10
    ↓
Data augmentation and normalization
    ↓
DataLoader
    ↓
Selected CNN architecture
    ↓
Forward propagation
    ↓
Cross-entropy loss
    ↓
Backpropagation
    ↓
Adam optimizer
    ↓
Evaluation on CIFAR-10 test set
```

The training function calculates the average training loss and classification accuracy for each epoch.

The evaluation function runs the model with gradient calculation disabled and reports the test loss and test accuracy.

## Model Architectures

### AlexNet

AlexNet was one of the most influential early deep convolutional neural networks.

The implementation in this repository is adapted for CIFAR-10. Because CIFAR-10 images are only `32 × 32`, the original large first convolution used for ImageNet is replaced with a smaller convolution:

```text
3 × 3 convolution
stride = 1
padding = 1
```

The network still follows the general AlexNet structure:

```text
Input
  ↓
Convolution + ReLU + Pooling
  ↓
Convolution + ReLU + Pooling
  ↓
Multiple convolutional layers
  ↓
Pooling
  ↓
Fully connected layers
  ↓
10-class output
```

### VGG

VGG demonstrated that deep convolutional networks can be constructed primarily from repeated small `3 × 3` convolutions.

The implementation uses reusable VGG blocks:

```text
Conv 3×3
   ↓
ReLU
   ↓
Conv 3×3
   ↓
ReLU
   ↓
MaxPool
```

Repeating the same simple building block makes the architecture deeper while keeping the design easy to understand.

### GoogLeNet

GoogLeNet introduces the Inception module.

Instead of applying only one convolution operation to an input feature map, an Inception block processes the same input through multiple parallel branches:

```text
                 ┌─ 1×1 Conv ────────────┐
                 │                        │
                 ├─ 1×1 → 3×3 Conv ──────┤
Input feature ───┤                        ├─ Concatenate
                 ├─ 1×1 → 5×5 Conv ──────┤
                 │                        │
                 └─ MaxPool → 1×1 Conv ──┘
```

The outputs from the four branches are concatenated along the channel dimension.

This allows the network to extract features at multiple receptive-field sizes within the same block.

### ResNet

ResNet introduces residual connections that allow information to bypass convolutional layers.

A basic residual block follows the structure:

```text
Input ────────────────────────────┐
  │                               │
  ↓                               │
3×3 Conv                          │
  ↓                               │
BatchNorm                         │
  ↓                               │
ReLU                              │
  ↓                               │
3×3 Conv                          │
  ↓                               │
BatchNorm                         │
  │                               │
  └──────────── Add ◀─────────────┘
                 ↓
                ReLU
```

When the spatial resolution or number of channels changes, a `1 × 1` convolution is applied to the shortcut branch so that the tensor shapes match.

The current configuration uses:

```python
((2, 64), (2, 128), (2, 256), (2, 512))
```

corresponding to the basic residual-block organization used by ResNet-18.

## Reproducibility

The project provides a random seed utility that configures:

- Python `random`
- NumPy
- PyTorch
- CUDA random number generation

The default seed is:

```text
2026
```

PyTorch deterministic algorithms are also enabled in the current implementation to improve reproducibility.

## Model Inspection

Each implemented architecture can be executed directly to inspect its structure using `torchinfo`.

For example:

```bash
python models/resnet.py
```

The model summary is generated using a dummy CIFAR-10 input:

```text
1 × 3 × 32 × 32
```

This makes it possible to inspect output shapes and parameter counts for individual layers.

## Current Scope

The current version focuses on implementing classic CNN architectures from scratch and providing a common CIFAR-10 training pipeline.

The project is still under development.

Possible future extensions include:

- DenseNet implementation
- Additional CNN architectures
- Training history recording
- Accuracy and loss visualization
- CSV logging
- Model checkpoint saving
- Best-model saving
- Learning-rate schedulers
- Additional optimizers
- CIFAR-100 support
- Parameter-count comparison
- Training-time comparison
- Automated benchmarking across models
- Confusion matrix visualization
- Class-wise accuracy
- Automated tests

The long-term goal is to provide a small and readable model zoo for studying and comparing the evolution of classic convolutional neural network architectures.

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.
