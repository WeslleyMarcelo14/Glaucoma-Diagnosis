# Glaucoma Diagnosis

A glaucoma diagnosis project using fundus images from the PAPILA dataset.

This project is divided into **two parts**:

| Part | Area | Context |
|------|------|---------|
| **Optic Disc Segmentation** | Computer Vision | Computer Vision Course - PPGI/UTFPR (Master's Program) |
| **Digital Image Processing** | DIP | Scientific Initiation - UTFPR (Undergraduate Research) |

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
│
├── src/                          # Source code
│   │
│   │ # ══════ COMPUTER VISION (Master's) ══════
│   ├── code.ipynb                # Main analysis
│   ├── experiment1.ipynb         # Experiment 1: U-Net Baseline
│   ├── experiment2.ipynb         # Experiment 2: CLAHE + Deep Supervision
│   ├── experiment3.ipynb         # Experiment 3: Attention U-Net
│   │
│   │ # ══════ DIP - SCIENTIFIC INITIATION ══════
│   └── dbscan.m                  # DBSCAN clustering
│
├── main.m                        # Main MATLAB script (SI)
├── utils/                        # Utility functions
├── results/                      # Output results
├── HelpCode/                     # Helper code
│
│ # ══════ DATA (not included in repository) ══════
├── FundusImages/                 # Fundus images
├── ExpertsSegmentations/         # Expert segmentations
└── ClinicalData/                 # Patient clinical data
```

---

## Part 1: Computer Vision (Master's Program - PPGI)

Optic disc segmentation using deep learning with different U-Net architectures.

### Experiments

| # | Model | Description |
|---|-------|-------------|
| 1 | **U-Net Baseline** | ResNet50 encoder (ImageNet), basic data augmentation, BCE + Dice loss |
| 2 | **U-Net Enhanced** | CLAHE, medical augmentations (Elastic, Grid, Optical), Deep Supervision, TTA |
| 3 | **Attention U-Net** | Attention Gates, SE blocks, ASPP, Focal Loss |

### Results

| Experiment | Model | Dice Score | IoU |
|:----------:|-------|:----------:|:---:|
| 1 | U-Net (ResNet50) | **0.9607** | **0.9252** |
| 2 | U-Net + Deep Supervision | 0.9604 | 0.9246 |
| 3 | Attention U-Net | 0.9499 | 0.9057 |

### Requirements (Python)

```
Python 3.x
PyTorch
segmentation-models-pytorch
Albumentations
NumPy, Pandas, OpenCV
Jupyter Notebook
```

---

## Part 2: Digital Image Processing (Scientific Initiation)

Fundus image analysis and processing using classical DIP techniques.

### Requirements (MATLAB)

```
MATLAB R2020a or later
Image Processing Toolbox
Statistics and Machine Learning Toolbox
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/glaucoma-diagnosis.git
cd glaucoma-diagnosis

# Install Python dependencies
pip install torch torchvision segmentation-models-pytorch albumentations opencv-python numpy pandas scikit-image jupyter

# Download the PAPILA dataset and extract to project root
```

---

## Usage

**Computer Vision (Python):**
```bash
jupyter notebook
# Open notebooks in src/
```

**DIP (MATLAB):**
```matlab
% Run main script
run('main.m')
```

---

## License

This project is for educational and research purposes.

---

## References

1. Kovalyk, O., Morales-Sánchez, J., Verdú-Monedero, R. et al. **PAPILA: Dataset for Glaucoma Assessment**. *Sci Data* 9, 291 (2022). [DOI](https://doi.org/10.1038/s41597-022-01388-1)
