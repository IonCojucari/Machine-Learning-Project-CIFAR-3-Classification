# Machine Learning Project: CIFAR-3 Classification

This project focuses on image classification using various machine learning techniques applied to the CIFAR-3 dataset, a subset of CIFAR-10. The dataset includes grayscale images of three classes: Automobile, Deer, and Horse. The project explores methods from dimensionality reduction to deep learning models, with the goal of developing effective classifiers.

## Project Overview

This repository contains:
- Python code implementing several machine learning models to classify images in the CIFAR-3 dataset.
- Data preparation and visualization techniques to better understand the dataset.
- Application of PCA for dimensionality reduction, traditional machine learning classifiers, and deep learning models including an MLP and a CNN.
  
### Dataset
The CIFAR-3 dataset consists of:
- `X_cifar_grayscale.npy`: Grayscale images, a tensor of shape (18000, 32, 32).
- `Y_cifar.npy`: Labels for each image, with classes represented as integers (0=Automobile, 1=Deer, 2=Horse).

### Requirements
- Python with dependencies: `numpy`, `matplotlib`, `scikit-learn`, and `tensorflow`.
- CIFAR-3 dataset files in the working directory.

## Code Description

### Data Loading and Preprocessing
- **Data loading**: The dataset is loaded and prepared for use in various models.
- **Preprocessing**: The grayscale images are flattened to a vector form (for PCA) or reshaped for CNN processing.

### Principal Component Analysis (PCA)
- **PCA Dimensionality Reduction**: Applied on grayscale images, with experiments using various numbers of components to understand data variance and feature reduction.
- **Reconstruction Visualization**: Reconstructed images from PCA to demonstrate data loss at different component levels.

### Supervised Machine Learning Models
- **Logistic Regression** and **Gaussian Naïve Bayes**: Basic classifiers trained on the grayscale data, with and without PCA-based dimensionality reduction.
- **Evaluation**: Accuracy scores are calculated on training and test sets.

### Deep Learning Models
- **Multilayer Perceptron (MLP)**: A fully connected neural network with hidden layers, dropout, and L2 regularization. Trained on PCA-reduced data.
- **Convolutional Neural Network (CNN)**: A convolutional model to handle the spatial structure in color images, with regularization and batch normalization for robust training.

### Training and Evaluation
Each model is evaluated on:
- **Training Accuracy** and **Test Accuracy**: Metrics are tracked to evaluate model performance.
- **Loss and Accuracy Plots**: Visualizations of training progress, with validation metrics included.

## How to Run the Code

1. **Install the dependencies** listed in `requirements.txt`.
2. **Place dataset files** (`X_cifar_grayscale.npy`, and `Y_cifar.npy`) in the working directory.
3. Run the script: 
   ```bash
   python script.py
