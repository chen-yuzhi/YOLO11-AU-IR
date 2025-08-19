# Detecting Infrared UAVs on Edge Devices through Lightweight Instance Segmentation

[![DOI](https://img.shields.io/badge/DOI-10.1371/journal.pone.0330074-blue.svg)](https://doi.org/10.1371/journal.pone.0330074)

## 📖 Introduction
**YOLO11-AU-IR** is a lightweight instance segmentation framework designed for **small, low-contrast UAV detection in infrared imagery** under real-time and resource-constrained conditions.  
Unlike conventional bounding-box methods, YOLO11-AU-IR generates **pixel-level segmentation masks**, capturing fine-grained UAV details and significantly improving accuracy in challenging infrared environments【Chen et al., 2025】.  

> 📄 This work has been published in:  
> **Chen Y, Sun H, Tian L, Yang Y, Wang S, Wang T (2025).**  
> *Detecting infrared UAVs on edge devices through lightweight instance segmentation.*  
> **PLOS ONE 20(8): e0330074.**  
> [https://doi.org/10.1371/journal.pone.0330074](https://doi.org/10.1371/journal.pone.0330074)

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/chen-yuzhi/YOLO11-AU-IR.git
cd YOLO11-AU-IR
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
python main.py train
```

## 📊 Dataset
AUVD-Seg300 is our curated IR UAV dataset with pixel-level segmentation masks. It captures various drone types and flight scenarios, ensuring robust training data for small-target detection in low-contrast IR conditions.
Upon acceptance and official publication of our manuscript, you will be able to download the dataset here, along with usage instructions.
Our infrared drone segmentation dataset contains **300 images**, divided into:
- **Training set**: 180 images
- **Validation set**: 60 images  
- **Test set**: 60 images

### Dataset Structure
```
AUVD-Seg300/
├── train/                       # Training set
│   ├── images/                  # Training set images
│   │   ├── 3.png
│   │   ├── 5.png
│   │   ├── 7.png
│   │   └── ...
│   └── labels/                  # Training set annotations
│       ├── 3.txt
│       ├── 5.txt
│       ├── 7.txt
│       └── ...
├── val/                         # Validation set
│   ├── images/                  # Validation set images
│   │   ├── 11.png
│   │   ├── 19.png
│   │   ├── 20.png
│   │   └── ...
│   └── labels/                  # Validation set annotations
│       ├── 11.txt
│       ├── 19.txt
│       ├── 20.txt
│       └── ...
└── test/                        # Test set
    ├── images/                  # Test set images
    │   ├── 1.png
    │   ├── 2.png
    │   ├── 4.png
    │   └── ...
    └── labels/                  # Test set annotations
        ├── 1.txt
        ├── 2.txt
        ├── 4.txt
        └── ...
```

### Download Dataset
📥 [Download AUVD-seg300.zip](https://1drv.ms/u/c/122bc9074aad62f0/ESP63poaAWhMjgQ7IuQjueIBkUIB8rXUeSY51fdJphfZcg?e=9NzObS)

## 🎯 Model Weights

Pre-trained model weights are available for download.

📥 [Download weights.pt](https://1drv.ms/u/c/122bc9074aad62f0/Edcnsy8x6BtBvrgPTmhMU6MBefCN0GpbcLXUQP0cNqFnGQ?e=Ans5O6)

## 🔮 Inference

Run inference on your infrared images:

```bash
python main.py predict --model ./weights.pt --source <your_infrared_image_path> --save
```

## 📌 Citation

If you use this work or dataset, please cite:

```bash
@article{chen2025yolo11auir,
  title     = {Detecting infrared UAVs on edge devices through lightweight instance segmentation},
  author    = {Chen, Yuzhi and Sun, Haoyue and Tian, Liang and Yang, Ye and Wang, Shenyang and Wang, Tianyou},
  journal   = {PLOS ONE},
  volume    = {20},
  number    = {8},
  pages     = {e0330074},
  year      = {2025},
  doi       = {10.1371/journal.pone.0330074}
}
```

---

⭐ If you find this project helpful, please consider giving it a star!

