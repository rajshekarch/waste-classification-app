# 🗑️ RealWaste Classification – DSCI 552 Final Project

## 📌 Overview

This project tackles **waste classification** using transfer learning on the **RealWaste dataset**.
The goal is to build a robust image classification pipeline to correctly identify waste categories (e.g., Cardboard, Plastic, Metal, Glass, etc.), supporting smarter recycling and waste management solutions.

We implemented and compared several deep learning architectures (ResNet50, ResNet101, EfficientNetB0, VGG16) with transfer learning and evaluated them on precision, recall, F1-score, and AUC metrics.

## 🚀 Features

* **Data Preprocessing**:

  * Custom pipeline using OpenCV for image resizing and augmentation
  * Train/test split with class balance preserved
* **Transfer Learning Models**:

  * ResNet50
  * ResNet101
  * EfficientNetB0
  * VGG16
* **Training Enhancements**:

  * EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
  * Batch Normalization & Dropout for regularization
* **Evaluation Metrics**:

  * Accuracy, Precision, Recall, F1-score
  * ROC curves & AUC for each class
  * Confusion Matrix visualization

## 📂 Dataset

* **RealWaste dataset** (organized into folders per class).
* Classes include: Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation.
* Images resized to **224 × 224** for uniform input across all models.


## 📊 Results (Highlights)

* **ResNet50 & EfficientNetB0** achieved the highest performance.
* Precision/Recall balanced across most classes.
* ROC curves showed strong separability for key categories.
* Confusion matrix highlighted overlap in visually similar categories (e.g., Paper vs Cardboard).


## 🛠️ Tech Stack

* **Languages/Frameworks**: Python 3.12, TensorFlow/Keras, OpenCV, NumPy, Pandas, Seaborn, Matplotlib
* **Models**: ResNet50, ResNet101, EfficientNetB0, VGG16 (transfer learning)
* **Training Environment**: GPU-accelerated (TensorFlow with CUDA/Metal support)


## 📌 Future Work

* Integrate dataset augmentation with real-world noise (lighting, rotations).
* Deploy as a **Flask/Streamlit web app** for live waste classification.
* Experiment with Vision Transformers (ViTs) for higher accuracy.

---

## 👨‍💻 Author

**Raja Shaker Chinthakindi**
Master’s in Applied Data Science @ University of Southern California (USC)
