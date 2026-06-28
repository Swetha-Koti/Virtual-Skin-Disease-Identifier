# 🩺 Virtual Skin Disease Identifier

A Deep Learning based Virtual Skin Disease Identifier that classifies skin lesions from dermoscopic images using a Custom Convolutional Neural Network (CNN). The model predicts one of seven skin diseases and provides a confidence score along with a clinical risk assessment (High, Moderate, or Low).

---

## Project Overview

Skin diseases are among the most common medical conditions worldwide. Early diagnosis is essential for effective treatment, but access to dermatologists is often limited.

This project uses Deep Learning to automate skin disease classification using dermoscopic images from the HAM10000 dataset.

The system predicts:

- Disease Type
- Prediction Confidence
- Clinical Risk Level

---

## Features

- Custom CNN Architecture
- Seven-Class Skin Disease Classification
- Confidence Score Prediction
- Risk Assessment Module
- TensorFlow/Keras Implementation
- HAM10000 Dataset
- Jupyter Notebook


---

## Dataset

HAM10000 Dataset

Contains:

- 10,015 Dermoscopic Images
- 7 Skin Disease Classes

Classes:

- Melanoma
- Basal Cell Carcinoma
- Actinic Keratoses
- Melanocytic Nevi
- Benign Keratosis
- Vascular Lesions
- Dermatofibroma

---

## Technologies

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
## Model Architecture

Custom Convolutional Neural Network (CNN)

Input

↓

Conv2D (32)

↓

MaxPooling

↓

Conv2D (64)

↓

MaxPooling

↓

Conv2D (128)

↓

MaxPooling

↓

Flatten

↓

Dense (128)

↓

Softmax Output

Optimizer:
Adam

Loss:
Sparse Categorical Crossentropy

Metric:
Accuracy

---

## Workflow

Dataset

↓

Image Preprocessing

↓

Label Encoding

↓

Resize Images (128×128)

↓

Normalization

↓

Data Augmentation

↓

CNN Training

↓

Prediction

↓

Risk Assessment

↓

Final Diagnosis

---

## Output

The model predicts

- Disease Name
- Confidence Score
- Risk Level

Example

Disease:
Melanocytic Nevi

Confidence:
34.07%

Risk:
LOW

---

## Installation

Clone Repository

```bash
git clone https://github.com/Swetha-Koti/Virtual-Skin-Disease-Identifier.git
```

Install Requirements

```bash
pip install -r requirements.txt
```

Run

```bash
jupyter notebook Skin_final_.ipynb
```



