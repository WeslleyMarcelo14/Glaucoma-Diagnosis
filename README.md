# Glaucoma Diagnosis

Projeto de diagnóstico de glaucoma utilizando imagens de fundo de olho do dataset PAPILA.

Este projeto é dividido em **duas partes**:

| Parte | Área | Contexto |
|-------|------|----------|
| **Segmentação do Disco Óptico** | Visão Computacional | Disciplina de Visão Computacional - PPGI/UTFPR (Mestrado) |
| **Processamento Digital de Imagens** | PDI | Iniciação Científica - UTFPR (Graduação) |

---

## Dataset

Este projeto utiliza o dataset **PAPILA** (PApular PILsen glaucomA):

| Recurso | Link |
|---------|------|
| Paper | [PAPILA: Dataset for Glaucoma Assessment](https://www.nature.com/articles/s41597-022-01388-1) |
| Download | [Figshare Repository](https://figshare.com/articles/dataset/PAPILA/14798004?file=35013982) |

---

## Estrutura do Projeto

```
glaucoma-diagnosis/
│
├── src/                          # Código fonte
│   │
│   │ # ══════ VISÃO COMPUTACIONAL (Mestrado) ══════
│   ├── code.ipynb                # Análise principal
│   ├── experiment1.ipynb         # Experimento 1: U-Net Baseline
│   ├── experiment2.ipynb         # Experimento 2: CLAHE + Deep Supervision
│   ├── experiment3.ipynb         # Experimento 3: Attention U-Net
│   │
│   │ # ══════ PDI - INICIAÇÃO CIENTÍFICA ══════
│   └── dbscan.m                  # Clustering DBSCAN
│
├── main.m                        # Script principal MATLAB (IC)
├── utils/                        # Funções utilitárias
├── results/                      # Resultados
├── HelpCode/                     # Código auxiliar
│
│ # ══════ DADOS (não incluídos no repositório) ══════
├── FundusImages/                 # Imagens de fundo de olho
├── ExpertsSegmentations/         # Segmentações dos especialistas
└── ClinicalData/                 # Dados clínicos dos pacientes
```

---

## Parte 1: Visão Computacional (Mestrado PPGI)

Segmentação do disco óptico utilizando deep learning com diferentes arquiteturas U-Net.

### Experimentos

| # | Modelo | Descrição |
|---|--------|-----------|
| 1 | **U-Net Baseline** | ResNet50 encoder (ImageNet), data augmentation básico, BCE + Dice loss |
| 2 | **U-Net Enhanced** | CLAHE, augmentations médicas (Elastic, Grid, Optical), Deep Supervision, TTA |
| 3 | **Attention U-Net** | Attention Gates, SE blocks, ASPP, Focal Loss |

### Resultados

| Experimento | Modelo | Dice Score | IoU |
|:-----------:|--------|:----------:|:---:|
| 1 | U-Net (ResNet50) | **0.9607** | **0.9252** |
| 2 | U-Net + Deep Supervision | 0.9604 | 0.9246 |
| 3 | Attention U-Net | 0.9499 | 0.9057 |

### Requisitos (Python)

```
Python 3.x
PyTorch
segmentation-models-pytorch
Albumentations
NumPy, Pandas, OpenCV
Jupyter Notebook
```

---

## Parte 2: Processamento Digital de Imagens (Iniciação Científica)

Análise e processamento de imagens de fundo de olho utilizando técnicas clássicas de PDI.

### Requisitos (MATLAB)

```
MATLAB R2020a ou superior
Image Processing Toolbox
Statistics and Machine Learning Toolbox
```

---

## Instalação

```bash
# Clonar repositório
git clone https://github.com/yourusername/glaucoma-diagnosis.git
cd glaucoma-diagnosis

# Instalar dependências Python
pip install torch torchvision segmentation-models-pytorch albumentations opencv-python numpy pandas scikit-image jupyter

# Baixar o dataset PAPILA e extrair na raiz do projeto
```

---

## Uso

**Visão Computacional (Python):**
```bash
jupyter notebook
# Abrir os notebooks em src/
```

**PDI (MATLAB):**
```matlab
% Executar o script principal
run('main.m')
```

---

## Licença

Projeto para fins educacionais e de pesquisa.

---

## Referências

1. Kovalyk, O., Morales-Sánchez, J., Verdú-Monedero, R. et al. **PAPILA: Dataset for Glaucoma Assessment**. *Sci Data* 9, 291 (2022). [DOI](https://doi.org/10.1038/s41597-022-01388-1)
