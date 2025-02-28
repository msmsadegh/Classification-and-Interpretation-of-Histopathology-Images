# Classification-and-Interpretation-of-Histopathology-Images

# Breast Cancer Classification Using CNNs and Ensemble Learning

## Overview
This repository contains the implementation of **breast cancer classification** using **Convolutional Neural Networks (CNNs)** and **ensemble learning** techniques. The study employs **EfficientNetV1 (b0-b4) and EfficientNetV2 (b0-b3)** architectures on the **BreakHis dataset**, leveraging **transfer learning, data augmentation, and Grad-CAM** for model interpretability.

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
├── notebooks/                    # Jupyter notebooks for exploration
├── README.md                      # Project documentation
└── requirements.txt                # Required dependencies
```

## Citation
If you use this repository, please cite our paper:
```
@article{azmoodeh-kalati2025classification,
  title={Classification and Interpretation of Histopathology Images: Leveraging Ensemble of EfficientNetV1 and EfficientNetV2 Models},
  author={Azmoodeh-Kalati, Mahdi and Shabani, Hasti and Maghareh, Mohammad Sadegh and Barzegar, Zeynab and Lashgari, Reza},
  year={2025}
}
```

## Contact
For questions or collaboration, reach out to: **mahdiazmoodeh95@gmail.com**, **r_lashgari@sbu.ac.ir**,**maghareh@aut.ac.ir**

---
This repository is developed for **academic research purposes** and aims to advance automated diagnostics in breast cancer classification.
---

## Notebook Descriptions
Below is a detailed list of the notebooks included in the `notebooks/` directory:

1. **cutfile.ipynb**  
   *Data splitting to train, validation, and test sets.*  
2. **breakhisefficientnetv1test1.ipynb**  
   *Training and learning curves for EfficientNetV1 models (variants b0–b4).*  
3. **breakhisefficientnetv2test1.ipynb**  
   *Training and learning curves for EfficientNetV2 model (variant b0-s).*  
4. **comparisonstained.ipynb**  
   *Comparison of performance for EfficientNetV1b0 on Vahadane stained images versus unstained images.*  
5. **multigradcam.ipynb**  
   *Generates Grad-CAM for individual models on the test set and merges them with their prediction probabilities.*  
6. **testv1v2report.ipynb**  
   *Analysis of test results including confusion matrix, misclassified images, and overall model performance.*  
7. **v1ensemble.ipynb**  
   *Creation and evaluation of an ensemble for EfficientNetV1 models along with a detailed report.*  
8. **v2ensemble.ipynb**  
   *Creation and evaluation of an ensemble for EfficientNetV2 models along with a detailed report.*  
9. **v1softhardvoting.ipynb**  
   *Prediction using unweighted averaging and hard (majority) voting for V1 models with performance results.*  
10. **v2softhardvoting.ipynb**  
    *Prediction using unweighted averaging and hard (majority) voting for V2 models with performance results.*  

---
