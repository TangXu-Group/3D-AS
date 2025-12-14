


# ID-Splat: Propagating Object Identities for Segmenting 3D Aerial-view Scenes

<div align="center">


Yijing Wang<sup>1</sup> · Xu Tang<sup>1*</sup> · Xiangrong Zhang<sup>1</sup> · Jingjing Ma<sup>1</sup>

<sup>1</sup>School of Artificial Intelligence, Xidian University, Xi'an, China  
<sup>*</sup>Corresponding author

</div>

---

## 📢 News

- **[November 2025]** Paper submitted to AAAI 2026 Oral
- **[December 2025]** Code repository initialized
- 🚧 **Code is currently under active development and will be fully released by the end of 2025**

---

## 📝 Abstract

High-resolution Earth Observation technologies present unprecedented opportunities for geospatial analysis, yet traditional 2D aerial-view semantic segmentation remains limited by its inability to model spatial relationships and handle object occlusions. While 3D Aerial-view Segmentation (3DAS) has emerged to address these limitations, existing methods predominantly rely on 2D discriminative models pre-trained on natural scenes, which struggle with aerial-view imagery due to significant domain discrepancies.

**ID-Splat** introduces a novel object-centric framework that directly leverages multi-view object identities without discriminative information to enhance 3D semantic understanding through:

1. **Mask-object Tracking**: Combines SAM and Point Tracking to establish robust and consistent object identities across multi-view aerial images
2. **Object Integration & Propagation**: Assigns identities to 3D Gaussian Splatting (3DGS) points, enabling complete 3D segmentation through semantic propagation

---

## 🏗️ Method Overview

<div align="center">
<img src="assets/framework.png" width="90%">
</div>

### Mask-object Tracking
- Establishes pixel correspondences using **Point Tracking**
- Generates object proposals with **SAM**
- Propagates object identities across views with **EOPS**

### Object Integration & Propagation
- Assigns object IDs to 3DGS points
- Generates pseudo-labels for unlabeled views
- Ensures complete coverage with **MSPS**

---

## 🚀 Getting Started

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.x (for GPU acceleration)
- GCC >= 11.4.0
- Git with submodule support

### Environment Setup

#### Step 1: Clone Repository with Submodules

```bash
# Clone repository with all submodules
git clone --recursive https://github.com/TangXu-Group/3D-AS.git
cd 3D-AS/ID-Splat

# If you already cloned without --recursive, initialize submodules:
git submodule update --init --recursive
```

#### Step 2: Create Conda Environment from YAML

```bash
# Create environment from the provided yaml file
conda env create -f environment.yaml

# Activate the environment
conda activate idsplat
```

#### Step 3: Install Submodule Dependencies

After creating the conda environment, install the required submodules:

```bash
mkdir submodules
git clone --recursive
git clone https://github.com/camenduru/simple-knn --recursive
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git --recursive
git clone https://github.com/rahul-goel/fused-ssim.git --recursive
git clone https://github.com/JoannaCCJH/gsplat.git --recursive
git clone https://github.com/minghanqin/segment-anything-langsplat.git --recursive
cd ..


# Install simple-knn
pip install submodules/simple-knn

# Install diff-gaussian-rasterization
pip install submodules/diff-gaussian-rasterization

# Install fused-ssim
pip install submodules/fused-ssim

# Install gsplat
pip install submodules/gsplat

# Install segment-anything-langsplat
pip install submodules/segment-anything-langsplat
```

---

## 📊 Dataset

Download the **3D-AS dataset** from:
- 🔗 [AliyunPan Link](https://www.alipan.com/t/pQPsyTyfLU7c2aju21UQ)

### Dataset Structure

Organize the downloaded data as follows:

```
data/
├── City/
│   ├── scene0/
│   │   ├── distorted/
│   │   ├── images/
│   │   ├── input/
│   │   ├── segmentation/
│   │   └── sparse/
│   ├── scene1/
│   └── scene2/
├── Country/
│   ├── scene0/
│   ├── scene1/
│   └── scene2/
└── Port/
    ├── scene0/
    ├── scene1/
    └── scene2/
```

### Dataset Details
- **Scenes**: 3 categories (City, Country, Port) × 3 scenes each
- **Images**: 70 multi-view aerial images per scene
- **Resolution**: ~1600×900 pixels 
- **Annotations**: Pixel-level segmentation for 6-8 semantic classes
- **Training**: Images 0, 7, 28 (3-view supervision)
- **Evaluation**: Images 14, 21, 35, 42, 49, 56, 63 (~2× downsampled)

---

## 🔗 Pretrained Models

| Model | Description | Download |
|-------|-------------|----------|
| 3DGS Initiation | Pre-trained 3D Gaussian Splatting | 🚧 Coming Soon |
| Mask-object IDs | Mask-object Tracking Results | 🚧 Coming Soon |

---

## 💻 Usage

> **Note**: Code is currently under development. The following commands are subject to change.

### 1. Mask-object Tracking

```bash
cd alltrack

# Extract SAM masks
bash extract_sam.sh

# Generate scene video
python generation_map4.py

# Extract point tracking results
bash extract_points_tracking.sh

# Extract SAM-object tracking results
bash sam_object_tracking.sh

# Transform object IDs (start from 0)
python transformer_idx.py
```

**Important**: Update the mask-object tracking results path in `./scene/dataset_readers.py` for the `objects_folder` parameter.

### 2. Training & Evaluation

```bash
# Run training
bash scripts/run_google.sh
```

### 3. Evaluation

Our evaluation follows the protocol from [3D-OVS](https://github.com/Kunhao-Liu/3D-OVS/tree/main).

Metrics:
- **mIoU** 
- **mAcc** 

---


## 🙏 Acknowledgements

This project builds upon the following excellent works:

- [**3D-OVS**](https://github.com/Kunhao-Liu/3D-OVS/tree/main) - 3D Open-Vocabulary Segmentation
- [**LangSplat**](https://github.com/minghanqin/LangSplat) - Language Gaussian Splatting
- [**Occam's LGS**](https://github.com/insait-institute/OccamLGS) - Efficient Language Gaussian Splatting
- [**SAM**](https://github.com/facebookresearch/segment-anything) - Segment Anything Model
- [**AllTracker**](https://github.com/aharley/alltracker) - Point Tracking

We thank the authors for their outstanding contributions to the community!

---


## 📧 Contact

For questions and discussions, please contact:

- **Yijing Wang**: [yijingwang@stu.xidian.edu.cn](yijingwang@stu.xidian.edu.cn) 


---

## 🔧 Submodules

This project relies on several external repositories as submodules:

| Submodule | Description | Source |
|-----------|-------------|--------|
| **simple-knn** | Fast K-Nearest Neighbors for 3DGS | [INRIA GitLab](https://gitlab.inria.fr/bkerbl/simple-knn.git) |
| **diff-gaussian-rasterization** | Differentiable Gaussian rasterization (dr_aa branch) | [GraphDeco INRIA](https://github.com/graphdeco-inria/diff-gaussian-rasterization.git) |
| **fused-ssim** | Fused SSIM implementation | [rahul-goel](https://github.com/rahul-goel/fused-ssim.git) |
| **gsplat** | Gaussian Splatting utilities (modified) | [JoannaCCJH](https://github.com/JoannaCCJH/gsplat.git) |
| **segment-anything-langsplat** | SAM integration for LangSplat | [minghanqin](https://github.com/minghanqin/segment-anything-langsplat) |
| **SIBR_viewers** | Structured IBR (Image-Based Rendering) viewer | [SIBR Core](https://gitlab.inria.fr/sibr/sibr_core.git) |

### Troubleshooting Submodule Installation

If you encounter issues with submodule installation:

```bash
# Update all submodules to latest
git submodule update --remote --merge

# Force reinstall a specific submodule
cd submodules/[submodule-name]
pip install -e .
cd ../..
```

---

## 🔄 Development Status

This repository is actively under development. Key milestones:

- [x] Paper submission 
- [x] Repository initialization
- [ ] Release Mask-object Tracking code
- [x] Release Object Integration & Propagation code
- [x] Release pretrained models
- [ ] Full code release 

Stay tuned for updates! ⭐ Star this repo to follow our progress.

---

</div>
