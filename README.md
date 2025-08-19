<a id="top"></a>

# Language · 语言
[English](#english-version) | [中文](#中文版)

---

<a id="english-version"></a>

# Detecting Infrared UAVs on Edge Devices through Lightweight Instance Segmentation (YOLO11-AU-IR)

[![DOI](https://img.shields.io/badge/DOI-10.1371/journal.pone.0330074-blue.svg)](https://doi.org/10.1371/journal.pone.0330074)

**YOLO11-AU-IR** targets **small, low-contrast** infrared UAVs with **real-time, robust segmentation on edge devices**.  
Compared with box-only detectors, it **directly predicts pixel-level instance masks**, preserving UAV contours more faithfully (Chen et al., 2025).

---

## 💡 Why it matters
- **Low contrast & high noise**: complex thermal backgrounds and blurry edges.
- **Tiny targets**: very few pixels; boxes easily include large background areas.
- **Diverse scenes**: sky/city/sea backgrounds prone to false alarms.
- **Edge constraints**: limited compute/memory; the network must be *light*.

**YOLO11-AU-IR** improves small-object separability via **lightweight instance segmentation**, while remaining deployable on embedded hardware.

---

## ✨ Key contributions
1. **Small-object instance segmentation** for low-contrast IR UAVs with stronger boundary expression.
2. **Edge-friendly**: fewer params, lower memory; fits Jetson-class devices.
3. **AUVD-Seg300 dataset**: 300 IR images with **pixel-accurate** masks.
4. **Reproducible pipeline**: open-source code, weights, and YOLO-style polygon mask format.

---

## 🚀 Quick Start

### 1) Clone
```bash
git clone https://github.com/chen-yuzhi/YOLO11-AU-IR.git
cd YOLO11-AU-IR
```

### 2) Install
```bash
pip install -r requirements.txt
```

### 3) Train
```bash
python main.py train
```

### 4) Inference
```bash
python main.py predict --model ./weights.pt --source <your_infrared_image_path> --save
```

---

## 📊 Dataset: AUVD-Seg300

**Model overview**  
![Fig3](https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0330074.g003)  
**multi-scale features** and **mask prediction** → use **polygon masks** rather than boxes only.

**AUVD-Seg300 samples & masks**  
![Fig2](https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0330074.g002)  
low contrast & diverse scenes; masks capture details (e.g., rotors/arms) missed by boxes.

**Directory structure**
```
AUVD-Seg300/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**YOLO polygon segmentation format**
- One instance per line:  
  `class_id x1 y1 x2 y2 x3 y3 ...` (coordinates normalized to 0–1)
- Example:
```text
0 0.421 0.318 0.447 0.319 0.455 0.336 0.430 0.339
```

**Download**
- 📥 Dataset archive: [AUVD-seg300.zip](https://1drv.ms/u/c/122bc9074aad62f0/ESP63poaAWhMjgQ7IuQjueIBkUIB8rXUeSY51fdJphfZcg?e=9NzObS)

**Start training**
```bash
python main.py train --data AUVD-Seg300
```

---

## 🧠 Method overview
- **From boxes to masks**: shape cues (arms/fuselage) reduce background interference.
- **Small-object friendly**: multi-scale & proper strides keep tiny UAVs resolvable.
- **Edge-first**: low params/memory/compute → embedded-friendly.

---

## 📈 Results & interpretation

**Qualitative comparisons**  
![Fig6](https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0330074.g006)  
Adheres closely to UAV boundaries under low contrast; alternatives are looser or miss detections.

**Accuracy–efficiency trade-offs**  
![Fig7](https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0330074.g007)  
Higher-quality masks at lower compute → suitable for **edge real-time**.

**Confusion matrix**  
![Fig9](https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0330074.g009)  
Strong main diagonal → robust UAV/background discrimination.

**Jetson TX2 tests**  
![Fig13](https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0330074.g013)  
Demonstrates **edge-side usability** and throughput.

---

## 🎯 Pretrained weights
- 📥 `weights.pt`: [Download](https://1drv.ms/u/c/122bc9074aad62f0/Edcnsy8x6BtBvrgPTmhMU6MBefCN0GpbcLXUQP0cNqFnGQ?e=Ans5O6)

---

## ⚠️ Limitations & future directions
- Extremely tiny targets.
- Sensor/background domain gaps.
- Heavy night-time humidity noise.

---

## 📌 Citation
```bibtex
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

**[中文 →](#中文版) · [↑ Back to top](#top)**

---

<a id="中文版"></a>

# 基于轻量级实例分割的红外无人机边端检测（YOLO11-AU-IR）

[![DOI](https://img.shields.io/badge/DOI-10.1371/journal.pone.0330074-blue.svg)](https://doi.org/10.1371/journal.pone.0330074)

**YOLO11-AU-IR** 面向**小尺寸、低对比度**红外无人机目标，追求**边端设备**上的实时可用与鲁棒分割。相比仅检测框的方法，**直接预测像素级实例掩膜（掩码）**，更好保留无人机轮廓细节（Chen 等, 2025）。

---

## 💡 为什么重要
- **对比度低、噪声高**：热辐射背景复杂，边缘模糊。
- **目标极小**：像素占比极低，检测框易包含大量背景。
- **场景多变**：天空/城市/海面等背景易产生误检。
- **端侧受限**：算力与内存有限，网络必须够“轻”。

**YOLO11-AU-IR** 以**轻量实例分割**提升小目标可分性，并兼顾端侧部署效率。

---

## ✨ 主要贡献
1. **面向小目标的实例分割**：对低对比度 IR UAV 具有更强边界表达。
2. **边端友好**：参数更少、显存占用低，适配 Jetson 等嵌入式设备。
3. **AUVD-Seg300 数据集**：300 张红外图像，**逐像素**精细标注。
4. **可复现管线**：开源代码、权重与 YOLO 风格分割标注格式。

---

## 🚀 快速开始

### 1）克隆
```bash
git clone https://github.com/chen-yuzhi/YOLO11-AU-IR.git
cd YOLO11-AU-IR
```

### 2）安装
```bash
pip install -r requirements.txt
```

### 3）训练
```bash
python main.py train
```

### 4）推理
```bash
python main.py predict --model ./weights.pt --source <你的红外图像路径> --save
```

---

## 📊 数据集：AUVD-Seg300

**模型总览**  
![图3](https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0330074.g003)  
强调**多尺度特征**与**掩膜预测**，因此数据使用**多边形掩膜**而非仅框。

**AUVD-Seg300 样例与掩膜**  
![图2](https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0330074.g002)  
低对比度、多场景；掩膜揭示了仅框方法易忽略的细节（如旋翼/机臂）。

**目录结构**
```
AUVD-Seg300/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**YOLO 分割标注格式（多边形）**
- 每行一个实例：  
  `class_id x1 y1 x2 y2 x3 y3 ...`（坐标归一化到 0–1）
- 示例：
```text
0 0.421 0.318 0.447 0.319 0.455 0.336 0.430 0.339
```

**下载**
- 📥 数据集压缩包：[AUVD-seg300.zip](https://1drv.ms/u/c/122bc9074aad62f0/ESP63poaAWhMjgQ7IuQjueIBkUIB8rXUeSY51fdJphfZcg?e=9NzObS)

**开始训练**
```bash
python main.py train --data AUVD-Seg300
```

---

## 🧠 方法概述
- **从框到掩膜**：用**实例掩膜**保留形状线索（机臂/机身等），降低背景干扰。
- **小目标友好**：多尺度与合适步长保持微小 UAV 的可分辨性。
- **端侧优先**：参数/显存/算力占用更低，利于嵌入式部署。

---

## 📈 结果与解读

**可视化对比**  
![图6](https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0330074.g006)  
在低对比度下仍能较好贴合 UAV 边界；相比之下，其他方法边界更松或漏检。

**精度与效率权衡**  
![图7](https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0330074.g007)  
在较低算力下获得更高质量掩膜，适配**边端实时**。

**混淆矩阵**  
![图9](https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0330074.g009)  
主对角线明显，说明对 UAV 与背景的区分更稳健。

**Jetson TX2 测试**  
![图13](https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0330074.g013)  
证明在嵌入式设备上的**端侧可用性**与吞吐表现。

---

## 🎯 预训练权重
- 📥 `weights.pt`：[下载链接](https://1drv.ms/u/c/122bc9074aad62f0/Edcnsy8x6BtBvrgPTmhMU6MBefCN0GpbcLXUQP0cNqFnGQ?e=Ans5O6)

---

## ⚠️ 已知局限与改进方向
- 极小目标仍困难。
- 不同红外传感器/背景的域差。
- 高湿度夜间噪声更重。

---

## 📌 引用
```bibtex
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

**[English →](#english-version) · [↑ 返回顶部](#top)**
