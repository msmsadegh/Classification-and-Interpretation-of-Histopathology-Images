# Classification and Interpretation of Histopathology Images  
### Leveraging Ensemble of EfficientNetV1 and EfficientNetV2 Models

**A Deep Learning-Based Approach for Breast Cancer Histopathology Image Analysis**

This repository contains the code, notebooks, and supplementary materials for our project on breast cancer histopathology image classification. Our approach leverages state-of-the-art EfficientNetV1 and EfficientNetV2 models, ensemble learning techniques, and Grad-CAM interpretability to achieve superior performance on the BreaKHis dataset. This work accompanies our paper submission to *Scientific Reports*.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Notebooks](#running-the-notebooks)
  - [Source Scripts and Command-Line Usage](#source-scripts-and-command-line-usage)
- [Key Results](#key-results)
- [Citation](#citation)
- [Contributions](#contributions)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
- [Future Work](#future-work)

---

## Project Overview
In this project, we address the critical task of accurately classifying histopathology images into benign and malignant categories. Breast cancer is one of the leading causes of cancer-related deaths among women, and early detection plays a crucial role in treatment planning and patient care.

**Key Highlights:**
- **Models:** We use EfficientNetV1 (variants b0–b4) and EfficientNetV2 (variant b0-s) architectures, known for their efficiency and strong performance in image classification tasks.
- **Ensemble Learning:** To further boost performance, ensemble methods such as unweighted averaging and hard (majority) voting are employed.
- **Interpretability:** Grad-CAM is applied to provide visual explanations of the model’s decisions, increasing transparency and trust in our predictions.
- **Dataset:** Our experiments are conducted on the widely used BreaKHis dataset, which comprises histopathological images of breast cancer at various magnifications.
- **Results:** Our best ensemble model achieved an accuracy of 99.58%, demonstrating the effectiveness of our approach.

---
---

## Installation

### Prerequisites
- **Python 3.7+**
- Recommended: A virtual environment (e.g., `venv` or `conda`)
- GPU support is recommended for training (CUDA-enabled GPU)

### Clone the Repository
```bash
git clone https://github.com/your-username/Classification-and-Interpretation-of-Histopathology-Images.git
cd Classification-and-Interpretation-of-Histopathology-Images

pip install -r requirements.txt


## 📊 Performance Summary
| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|------------|------------|------------|
| EfficientNetV1-B4 | 99.07% | 99.38% | 99.26% | 99.32% |
| EfficientNetV2-B1 | 99.32% | 99.51% | 99.51% | 99.51% |
| **V1 Ensemble** | **99.49%** | **99.51%** | **99.75%** | **99.63%** |
| **V2 Ensemble** | **99.58%** | **99.88%** | **99.51%** | **99.69%** |


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