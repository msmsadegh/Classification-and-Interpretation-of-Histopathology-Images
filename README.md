# Classification-and-Interpretation-of-Histopathology-Images

# Breast Cancer Classification Using CNNs and Ensemble Learning

## Overview
This repository contains the implementation of **breast cancer classification** using **Convolutional Neural Networks (CNNs)** and **ensemble learning** techniques. The study employs **EfficientNetV1 (b0-b4) and EfficientNetV2 (b0-b3)** architectures on the **BreakHis dataset**, leveraging **transfer learning, data augmentation, and Grad-CAM** for model interpretability.

## Branches
We provide different branches corresponding to the methods explored in our study:

- **`cnn_classification`**: Implements CNN-based classification using EfficientNet models.
- **`cnn_ensemble`**: Incorporates unweighted averaging and majority voting for ensemble learning.
- **`multi_cnn_mlp`**: Combines multiple CNN models with a Multi-Layer Perceptron (MLP) for enhanced classification accuracy.

## Features
- **Dataset**: Utilizes the **BreakHis dataset** for binary classification of histopathological images.
- **CNN Models**: EfficientNetV1 & EfficientNetV2 architectures.
- **Transfer Learning**: Pretrained CNN models fine-tuned for breast tissue classification.
- **Data Augmentation**: Applied for better generalization and performance.
- **Model Interpretability**: **Grad-CAM** is used to visualize critical regions influencing predictions.
- **Ensemble Learning**: Majority voting & unweighted averaging to improve prediction robustness.

## Getting Started
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install tensorflow keras numpy pandas matplotlib opencv-python scikit-learn torch torchvision torchaudio albumentations
```

### Clone the Repository
```bash
git clone https://github.com/your_username/breast_cancer_cnn.git
cd breast_cancer_cnn
```

### Dataset Setup
Download the **BreakHis dataset** from [here](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database/) and place it in the `data/` directory.

### Training the Model
To train a CNN model, run:
```bash
python train.py --model EfficientNetV2 --epochs 50 --batch_size 32
```

To train the ensemble model:
```bash
python ensemble_training.py
```

## Results
The most effective ensemble model, using **majority voting from EfficientNetV2 (b0-S)**, achieved **99.62% accuracy** on the **BreakHis dataset**, surpassing conventional CNN-based approaches.

## Repository Structure
```
├── data/                         # BreakHis dataset (to be downloaded manually)
├── models/                       # Saved model checkpoints
├── notebooks/                    # Jupyter notebooks for exploration
├── src/
│   ├── train.py                  # CNN training script
│   ├── ensemble_training.py       # Ensemble training script
│   ├── utils.py                   # Utility functions
│   ├── grad_cam.py                # Grad-CAM visualization
├── README.md                      # Project documentation
└── requirements.txt                # Required dependencies
```

## Citation
If you use this repository, please cite our paper:
```
@article{your_paper,
  title={Breast Cancer Classification Using CNNs and Ensemble Learning},
  author={Mahdi Azmoodeh Kalati, Hasti Shabani, Mohammad Sadegh Maghareh, Zeynab Barzegar, Reza Lashgari},
  year={2025},
  journal={Scientific Reports}
}
```

## Contact
For questions or collaboration, reach out to: **mahdiazmoodeh95@gmail.com**, **r_lashgari@sbu.ac.ir**,**maghareh@aut.ac.ir**

---
This repository is developed for **academic research purposes** and aims to advance automated diagnostics in breast cancer classification.
