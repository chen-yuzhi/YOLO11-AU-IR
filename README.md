# Detecting infrared UAVs on edge devices through lightweight instance segmentation


## Introduction
YOLO11-AU-IR is a lightweight instance segmentation framework aimed at addressing small, low-contrast UAV detection in infrared imagery under real-time and resource-constrained conditions. Unlike standard bounding-box-based methods, our approach leverages pixel-level segmentation masks to capture fine-grained details of small IR targets, thus improving detection accuracy even in challenging environments.




## 🚀 Quick Start

### Clone the Repository
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


---

⭐ If you find this project helpful, please consider giving it a star!
