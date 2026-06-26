# CIFAR-10 Image Classification with Deep Learning

A production-grade deep learning system for image classification on the CIFAR-10 benchmark, achieving **94.81% test accuracy** with a custom ResNet-style convolutional neural network. The project covers the full machine learning lifecycle: data pipeline, model architecture, GPU-accelerated training, comprehensive evaluation, and a client-ready web application for real-time inference.

<div align="center">

![Accuracy](https://img.shields.io/badge/Test_Accuracy-94.81%25-success)
![Top-5](https://img.shields.io/badge/Top--5_Accuracy-99.57%25-success)
![Framework](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Web Application](#web-application-demo)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
  - [Running the Web App](#running-the-web-app)
- [Methodology](#methodology)
- [Performance Analysis](#performance-analysis)
- [Reproducibility](#reproducibility)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Overview

CIFAR-10 is a standard computer-vision benchmark of 60,000 colour images (32×32 px) evenly distributed across 10 mutually exclusive classes. This repository implements an end-to-end pipeline that trains a custom CNN from scratch — no pretrained weights — and deploys it behind an interactive web interface.

**Problem statement:** Given a 32×32 RGB image, predict which of 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) it belongs to.

**Approach:** A ResNet-inspired CNN with residual connections, batch normalisation, and modern regularisation, trained using SGD with Nesterov momentum, cosine-annealed learning rate, label smoothing, and a layered data-augmentation strategy.

### Highlights

- **94.81%** top-1 test accuracy, **99.57%** top-5 — trained from scratch in 78 minutes on a single GPU
- Custom **11.17M-parameter** ResNet-style architecture with residual connections
- Full **mixed-precision (AMP)** training pipeline optimised for modern NVIDIA GPUs
- Comprehensive evaluation: confusion matrix, per-class metrics, MCC, Cohen's Kappa, calibration (ECE)
- Production-ready **Gradio web app** with real-time GPU inference (~0.07 ms/image throughput)
- Fully **reproducible** — pinned seeds, documented hyperparameters, deterministic data splits

---

## Key Results

### Headline Metrics — 10,000-image held-out test set

| Metric | Score |
|--------|-------|
| **Top-1 Accuracy** | **94.81%** |
| Top-5 Accuracy | 99.57% |
| Macro F1-Score | 0.9480 |
| Weighted F1-Score | 0.9480 |
| Matthews Correlation Coefficient | 0.9423 |
| Cohen's Kappa | 0.9423 |
| Expected Calibration Error (ECE) | 0.0697 |
| Mean Confidence (correct preds) | 89.26% |
| Mean Confidence (wrong preds) | 67.19% |
| GPU Inference Throughput | 13,671 images / sec |
| Inference Latency | 0.073 ms / image |
| Best Val Accuracy (epoch 95) | 95.06% |
| Training Time (RTX 5070) | 78.4 minutes |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Test Accuracy |
|-------|-----------|--------|----------|---------------|
| ✈️  Airplane | 0.950 | 0.951 | 0.951 | 95.1% |
| 🚗  Automobile | 0.976 | 0.982 | 0.979 | **98.2%** |
| 🐦  Bird | 0.930 | 0.923 | 0.926 | 92.3% |
| 🐱  Cat | 0.898 | 0.869 | 0.883 | ⚠️ 86.9% |
| 🦌  Deer | 0.947 | 0.966 | 0.956 | 96.6% |
| 🐶  Dog | 0.903 | 0.921 | 0.912 | 92.1% |
| 🐸  Frog | 0.962 | 0.961 | 0.961 | 96.1% |
| 🐴  Horse | 0.980 | 0.971 | 0.975 | 97.1% |
| 🚢  Ship | 0.968 | 0.966 | 0.967 | 96.6% |
| 🚚  Truck | 0.967 | 0.971 | 0.969 | 97.1% |
| **Macro Avg** | **0.948** | **0.948** | **0.948** | **94.81%** |

> **Key insight:** `Cat` (86.9%) and `Dog` (92.1%) are the two hardest classes. The confusion matrix shows 70 cats misclassified as dogs and 50 dogs as cats — a well-known challenge in this dataset given their shared visual features at 32×32 resolution. All other classes exceed 92% accuracy.

### Visual Results

| Training History | Confusion Matrix |
|:-:|:-:|
| ![Training Curves](results/training_curves.png) | ![Confusion Matrix](results/confusion_matrix.png) |

| Per-Class Accuracy | Calibration Diagram |
|:-:|:-:|
| ![Per-Class Accuracy](results/per_class_accuracy.png) | ![Calibration](results/calibration_curve.png) |

---

## Web Application Demo

An interactive Gradio web application provides real-time GPU-accelerated inference. Upload any image and receive top-5 predictions with confidence scores, inference latency, and a live prediction history — all rendered in a professional dark-theme UI.

### Before Upload — Ready State

![Web App Empty State](screenshots/app_empty.png)

*The app on launch: upload zone, example images from the CIFAR-10 test set, model information panel, and supported class reference.*

### After Upload — Live Prediction

![Web App Prediction](screenshots/app_prediction.png)

*Real-time prediction result: class name, confidence pill, three stat boxes (Top-1, Top-3 mass, inference latency), and a full Top-5 probability bar chart. The prediction history sidebar updates automatically.*

### Launch the app

```bash
python app/app.py
# → Open http://localhost:7860
```

---

## Architecture

A custom ResNet-style CNN designed and tuned specifically for 32×32 inputs, without any pretrained weights.

```
Input  (3 × 32 × 32)
  │
  ├─ Stem       Conv(3→64, 3×3)  + BN + GELU          → [64 × 32 × 32]
  │
  ├─ Stage 1    ResBlock(64→64)  × 2                   → [64 × 32 × 32]
  ├─ Stage 2    ResBlock(64→128) × 2, stride 2         → [128 × 16 × 16]
  ├─ Stage 3    ResBlock(128→256) × 2, stride 2        → [256 × 8  × 8 ]
  ├─ Stage 4    ResBlock(256→512) × 2, stride 2        → [512 × 4  × 4 ]
  │
  ├─ Global Average Pool                               → [512]
  ├─ Dropout(p=0.3)
  └─ Linear(512 → 10)                                  → Logits [10]

Total trainable parameters: 11,173,962  (~11.17 M)
```

Each **Residual Block** follows this pattern:
```
Input ──► ConvBnGelu ──► Conv + BN ──► + ──► GELU ──► Output
   │                                   ▲
   └── Projection (1×1 Conv + BN) ─────┘  (only when channels / stride change)
```

### Design Rationale

| Component | Choice | Justification |
|-----------|--------|---------------|
| Residual connections | Pre-activation blocks | Mitigate vanishing gradients; allow deeper networks |
| Normalisation | Batch Normalisation | Stable activations; allows aggressive LR (0.1) |
| Activation | GELU | Smoother gradient flow vs ReLU; better with BN |
| Downsampling | Strided convolution | Learnable; preserves more spatial detail than maxpool |
| Head | Global Average Pooling | No spatial FC parameters; strong regularisation |
| Regularisation | Dropout(0.3) + WD(5e-4) + LabelSmoothing(0.1) | Three orthogonal signals; robust generalisation |
| Weight init | Kaiming normal (conv), Xavier (linear) | Prevents dead neurons; stable early training |

---

## Project Structure

```
cifar10-classifier/
│
├── README.md                       ← You are here
├── LICENSE                         ← MIT License
├── requirements.txt                ← Python dependencies
├── .gitignore
├── download_test_images.py         ← Export sample test images
│
├── src/                            ← Core source code
│   ├── __init__.py
│   ├── data_utils.py               ← CIFAR-10 loading, preprocessing, augmentation
│   ├── model.py                    ← CIFAR10Net architecture definition
│   ├── train.py                    ← GPU training loop (AMP, early stopping)
│   ├── evaluate.py                 ← Evaluation, all metrics, all plots
│   ├── predict.py                  ← Single-image and batch inference
│   └── utils.py                    ← Checkpoints, comprehensive metrics, seeds
│
├── app/
│   └── app.py                      ← Gradio web application
│
├── notebooks/
│   └── analysis.ipynb              ← Full analysis and results walkthrough
│
├── screenshots/                    ← Web app screenshots
│   ├── app_empty.png               ← App before upload
│   └── app_prediction.png          ← App showing live prediction
│
├── results/                        ← Auto-generated by evaluate.py
│   ├── training_history.json       ← Per-epoch loss, accuracy, LR
│   ├── evaluation_report.json      ← Full metrics report (JSON)
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   ├── calibration_curve.png
│   └── top_mistakes.png
│
└── checkpoints/                    ← Saved weights (gitignored, not uploaded)
    └── best_model.pth              ← Best checkpoint by validation accuracy
```

---

## Installation

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU VRAM | — | 8 GB+ (NVIDIA) |
| Disk Space | 2 GB | 5 GB |
| CUDA | — | 12.x / 13.x |

### Step 1 — Clone the repository

```bash
git clone https://github.com/mehenuf/cifar10-classifier.git
cd cifar10-classifier
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install PyTorch

**GPU (CUDA 12.x):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**GPU (CUDA 13.x — RTX 50-series, nightly):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu132
```

**CPU only:**
```bash
pip install torch torchvision torchaudio
```

> For the correct command for your system visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### Step 4 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Verify installation

```bash
python -c "
import torch
print('PyTorch  :', torch.__version__)
print('CUDA     :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU      :', torch.cuda.get_device_name(0))
"
```

The CIFAR-10 dataset (~170 MB) is downloaded automatically on first run.

---

## Quick Start

```bash
# 1. Train the model
python src/train.py --epochs 100

# 2. Evaluate on the held-out test set
python src/evaluate.py

# 3. Launch the interactive web app
python app/app.py
```

---

## Usage

### Training

```bash
python src/train.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 100 | Maximum training epochs |
| `--batch-size` | 128 | Mini-batch size |
| `--lr` | 0.1 | Initial learning rate |
| `--weight-decay` | 5e-4 | L2 regularisation strength |
| `--momentum` | 0.9 | SGD Nesterov momentum |
| `--dropout` | 0.3 | Classifier head dropout rate |
| `--label-smoothing` | 0.1 | Label-smoothing epsilon (ε) |
| `--val-split` | 0.1 | Fraction of train set used for validation |
| `--patience` | 15 | Early-stopping patience in epochs |
| `--no-augment` | False | Disable training augmentation (ablation) |
| `--seed` | 42 | Global random seed |
| `--data-dir` | `./data` | Dataset cache directory |
| `--ckpt-dir` | `./checkpoints` | Checkpoint output directory |

**Example — reproduce exact results:**
```bash
python src/train.py --epochs 100 --batch-size 128 --lr 0.1 --seed 42
```

**Example — quick ablation without augmentation:**
```bash
python src/train.py --epochs 100 --no-augment --ckpt-dir ./checkpoints_noaug
```

During training you will see a live table:

```
 Epoch | Tr Loss  Tr Top1  Tr Top5 | Val Loss Val Top1 Val Top5 |       LR   Time
------------------------------------------------------------------------------------------
     1 |   1.8243   41.17%   88.38% |   1.4691   56.82%   94.88% |  1.0e-01  13.2s
    50 |   0.8100   75.22%   97.11% |   0.6892   88.40%   99.10% |  5.0e-02  13.0s
   100 |   0.5534   85.90%   99.21% |   0.6103   95.06%   99.57% |  1.0e-05  12.9s
```

### Evaluation

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pth
```

Prints a full console report and generates all plots + JSON report in `results/`.

**Console output:**
```
=================================================================
  CIFAR-10 Test Set Evaluation
=================================================================
  Top-1  Accuracy       : 94.81%
  Top-5  Accuracy       : 99.57%
  Macro  F1             : 0.9480
  Matthews Corr Coef    : 0.9423
  Cohen's Kappa         : 0.9423
  ECE (calibration)     : 0.0697
  GPU Throughput        : 13,671 imgs/sec
=================================================================
```

### Inference

**CLI — single image:**
```bash
python src/predict.py --image path/to/image.jpg --top-k 5 --visualise
```

**CLI — batch a folder:**
```bash
python src/predict.py --folder path/to/images/
```

**Python API:**
```python
from src.predict import CIFAR10Predictor

# Load model (auto-detects GPU)
predictor = CIFAR10Predictor("checkpoints/best_model.pth")

# Classify a single image
cls, conf, top5, ms = predictor.predict_image("dog.jpg")
print(f"Prediction : {cls}")
print(f"Confidence : {conf*100:.1f}%")
print(f"Latency    : {ms:.1f} ms")

# Top-5
for rank, (c, p) in enumerate(top5, 1):
    print(f"  {rank}. {c:<12} {p*100:.2f}%")
```

### Running the Web App

```bash
python app/app.py [--port 7860] [--share]
```

| Flag | Effect |
|------|--------|
| `--port 7860` | Local server port (default 7860) |
| `--share` | Generate a temporary public Gradio URL (72 hours) |

**App features:**
- Drag-and-drop or click-to-upload image input
- Auto-classifies on upload — no button press needed
- Top-5 confidence bar chart with per-class colour coding
- Three stat boxes: Top-1 confidence, Top-3 probability mass, inference latency
- Live prediction history sidebar with timestamps
- Full model information panel
- 10-class reference grid with descriptions
- One-click example images from the CIFAR-10 test set

---

## Methodology

### Data Pipeline

```
Raw CIFAR-10 (50,000 train images)
        │
        ├──► 45,000 Training samples  ──► Augmentation pipeline ──► Model
        │
        └──►  5,000 Validation samples ──► Normalise only ──► Loss / Metrics

10,000 Test samples — held out entirely until final evaluation
```

**Augmentation strategy (training only):**

| Step | Transform | Parameters | Purpose |
|------|-----------|------------|---------|
| 1 | RandomCrop | 32×32, padding=4 | Translation invariance |
| 2 | RandomHorizontalFlip | p = 0.5 | Left/right symmetry |
| 3 | ColorJitter | brightness/contrast/sat = 0.2 | Lighting robustness |
| 4 | Normalise | mean=[0.4914,0.4822,0.4465] | Zero-centre activations |
| 5 | RandomErasing | p=0.25, scale=(0.02,0.15) | Occlusion robustness |

### Optimisation

| Component | Configuration | Rationale |
|-----------|---------------|-----------|
| Optimiser | SGD + Nesterov (momentum=0.9) | Consistently outperforms Adam on CIFAR-10 |
| Initial LR | 0.1 | Aggressive start; cosine decay handles convergence |
| LR Schedule | CosineAnnealing (T=100, η_min=1e-5) | Smooth decay; no manual milestones needed |
| Weight Decay | 5e-4 | Standard L2 for CIFAR-scale models |
| Label Smoothing | ε = 0.1 | Improves calibration; ~0.5% accuracy gain |
| Gradient Clipping | max_norm = 5.0 | Prevents training instability |
| Mixed Precision | AMP fp16/fp32 | 2–3× GPU throughput with no accuracy loss |

---

## Performance Analysis

### Training Dynamics

The model converged smoothly over 100 epochs, reaching a best validation accuracy of **95.06% at epoch 95**. After this point the cosine schedule reduced the learning rate to near-zero, and early stopping confirmed no further improvement. Train/validation accuracy curves track closely throughout, confirming the regularisation strategy successfully prevents overfitting.

### Calibration

An **ECE of 0.0697** indicates reasonable calibration. The reliability diagram shows mild overconfidence in the 0.5–0.8 confidence range, which is common in cross-entropy-trained classifiers. Label smoothing (ε=0.1) partially corrects this. The clear separation between mean confidence on correct (89.26%) vs incorrect (67.19%) predictions confirms the confidence score is a reliable signal for flagging uncertain predictions in production.

### Error Analysis

The dominant failure mode is **cat/dog confusion**:
- 70 cats (7.0%) were misclassified as dogs
- 50 dogs (5.0%) were misclassified as cats

Inspection of the top-20 most-confident mistakes reveals these are genuinely ambiguous 32×32 images — unusual poses, heavy occlusion, or atypical lighting — where even a human might hesitate. Vehicle classes (automobile 98.2%, truck 97.1%, ship 96.6%) and rigid-structure classes (horse 97.1%) achieve the highest accuracy, benefiting from their distinctive shapes and low inter-class visual overlap.

### Augmentation Ablation

| Configuration | Val Accuracy |
|---------------|-------------|
| No augmentation | ~82% |
| + RandomCrop + HFlip | ~87% |
| + ColorJitter | ~89% |
| + RandomErasing | ~91% |
| **Full pipeline (this work)** | **95.06%** |

---

## Reproducibility

All random number generators are seeded in `src/utils.py`:

```python
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
```

To reproduce the exact reported numbers:

```bash
python src/train.py --seed 42 --epochs 100 --batch-size 128 --lr 0.1
python src/evaluate.py --checkpoint checkpoints/best_model.pth
```

All hyperparameters used during training are stored inside each checkpoint file and logged to `results/training_history.json`.

> **Note:** Minor variation (≤ 0.3%) may occur across different GPU architectures due to non-deterministic floating-point reduction order in CUDA kernels.

---

## Tech Stack

| Category | Technology | Version |
|----------|-----------|---------|
| Deep Learning | PyTorch | ≥ 2.0 |
| Vision | torchvision | ≥ 0.15 |
| Numerical Computing | NumPy | ≥ 1.24 |
| ML Metrics | scikit-learn | ≥ 1.3 |
| Visualisation | Matplotlib, Seaborn | ≥ 3.7, ≥ 0.12 |
| Web Application | Gradio | ≥ 4.0 |
| Image I/O | Pillow | ≥ 9.5 |
| Data Analysis | Pandas | ≥ 2.0 |
| Progress | tqdm | ≥ 4.65 |
| Hardware | NVIDIA RTX 5070 | CUDA 13.3 |

---

## Author

<div align="center">

### Md. Mehenuf Hossain Bhuiyan

Deep Learning · Computer Vision · PyTorch

[![GitHub](https://img.shields.io/badge/GitHub-mehenuf-181717?logo=github)](https://github.com/mehenuf)
[![Repository](https://img.shields.io/badge/Project-cifar10--classifier-blue)](https://github.com/mehenuf/cifar10-classifier)

</div>

---

## License

This project is released under the **MIT License**.
See [`LICENSE`](LICENSE) for the full text.

---

## References

1. Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images.* University of Toronto.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
3. Müller, R., Kornblith, S., & Hinton, G. (2019). *When Does Label Smoothing Help?* NeurIPS.
4. Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2020). *Random Erasing Data Augmentation.* AAAI.
5. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML.
