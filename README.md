# Glaucoma Diagnosis

A machine learning project for glaucoma diagnosis using fundus images from the PAPILA dataset.

> **Scientific Initiation (Undergraduate Research)**: The digital image processing (DIP) component belongs to this research.
>
> **Master's Program**: The deep learning and computer vision algorithms were developed for the Computer Vision course of PPGI (Graduate Program in Informatics) at UTFPR (Federal University of Technology - Paraná).

---

## Overview

This project implements algorithms for glaucoma detection and classification using retinal fundus images. It utilizes expert segmentations of the optic disc and cup regions along with clinical patient data.

The main focus is on **optic disc segmentation** using deep learning techniques, comparing different U-Net architectures and training strategies.

---

## Dataset

This project uses the **PAPILA** (PApular PILsen glaucomA) dataset:

| Resource | Link |
|----------|------|
| Paper | [PAPILA: Dataset for Glaucoma Assessment](https://www.nature.com/articles/s41597-022-01388-1) |
| Download | [Figshare Repository](https://figshare.com/articles/dataset/PAPILA/14798004?file=35013982) |

---

## Project Structure

```
glaucoma-diagnosis/
├── ClinicalData/             # Patient clinical data (Excel files)
├── ExpertsSegmentations/     # Expert annotations
│   ├── Contours/             # Disc and cup contour coordinates
│   └── ImagesWithContours/   # Annotated images
├── FundusImages/             # Retinal fundus images (JPG)
├── HelpCode/                 # Helper code and examples
├── src/                      # Source code and experiments
│   ├── code.ipynb            # Main analysis notebook
│   ├── experiment1.ipynb     # Experiment 1: Baseline U-Net
│   ├── experiment2.ipynb     # Experiment 2: CLAHE + Deep Supervision
│   ├── experiment3.ipynb     # Experiment 3: Attention U-Net
│   └── dbscan.m              # DBSCAN clustering (MATLAB)
├── results/                  # Output results
├── utils/                    # Utility functions
└── main.m                    # Main MATLAB script
```

> **Note**: The `FundusImages/`, `ExpertsSegmentations/`, and `ClinicalData/` folders are not included in this repository due to size. Download the PAPILA dataset from the link above.

---

## Experiments

| # | Model | Description |
|---|-------|-------------|
| 1 | **Baseline U-Net** | U-Net with ResNet50 encoder (ImageNet pretrained), basic data augmentation, BCE + Dice loss |
| 2 | **Enhanced U-Net** | CLAHE preprocessing, advanced medical augmentations (Elastic, Grid, Optical distortions), Deep Supervision, TTA |
| 3 | **Attention U-Net** | Attention Gates, SE blocks, ASPP at bottleneck, Focal Loss |

---

## Results

| Experiment | Model | Dice Score | IoU |
|:----------:|-------|:----------:|:---:|
| 1 | U-Net (ResNet50) | **0.9607** | **0.9252** |
| 2 | U-Net + Deep Supervision | 0.9604 | 0.9246 |
| 3 | Attention U-Net | 0.9499 | 0.9057 |

---

## Requirements

```
Python 3.x
PyTorch
segmentation-models-pytorch
Albumentations
NumPy
Pandas
Scikit-image
OpenCV
Jupyter Notebook
MATLAB (optional, for DIP experiments)
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/glaucoma-diagnosis.git
cd glaucoma-diagnosis

# Install dependencies
pip install torch torchvision segmentation-models-pytorch albumentations opencv-python numpy pandas scikit-image jupyter

# Download the PAPILA dataset and extract to project root
```

---

## Usage

```bash
# Start Jupyter Notebook
jupyter notebook

# Navigate to src/ and open experiment notebooks
```

---

## License

This project is for educational and research purposes.

---

## References

1. Kovalyk, O., Morales-Sánchez, J., Verdú-Monedero, R. et al. **PAPILA: Dataset for Glaucoma Assessment**. *Sci Data* 9, 291 (2022). [DOI](https://doi.org/10.1038/s41597-022-01388-1)
