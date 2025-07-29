# 🔍🖼️ Content-Based Image Retrieval Using Hypergraph Ranking 

This project implements a **Content-Based Image Retrieval (CBIR)** system using **unsupervised learning** and **hypergraph-based manifold ranking**.

## 📋 Overview

Given a large dataset of images, the system allows users to retrieve the most visually similar images to a query image using deep learning for feature extraction and a hypergraph model to compute image similarities.

### ✨ Key Features
- Uses **ResNet-50**, a pre-trained Convolutional Neural Network, for **feature extraction**.
- Constructs a **hypergraph** to model complex relationships between images.
- Implements a **custom ranking algorithm** based on **unsupervised hypergraph-based manifold learning**.
- Efficient **image retrieval** with visual similarity ranking.
- Measures retrieval **accuracy** using **Mean Squared Error (MSE)**.

## 🗂️ Dataset
- **Corel5K** dataset (18,500 images, 128MB total size).
- Only images are used—**no labels or annotations**—ensuring an **unsupervised** approach.

## 🧰 Dependencies
- Python
- NumPy
- TensorFlow
- Scikit-learn
- Matplotlib
- SciPy

## ▶️ Running the Project

- Clone the repository to your local machine. The `Corel5K` dataset is included by default, but you can replace it with another dataset if you prefer.
- Run `setup.py` once to generate all necessary output files for the image search.
- Run `main.py` with an image—either from the dataset or your own—to find and display the most similar images.

## 📖 Citation
Based on the algorithm presented in:

> Multimedia Retrieval through Unsupervised Hypergraph-based Manifold Ranking, IEEE Transactions on Image Processing, Vol. 28, No. 12, December 2019.


