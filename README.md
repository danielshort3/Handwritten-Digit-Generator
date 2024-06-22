# Handwritten Digit Generator

This repository contains a project aimed at training a Variational Autoencoder (VAE) on the MNIST dataset. The project includes data preparation, model definition, training and evaluation procedures, and various visualizations and analyses of the results.

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Data Analysis](#data-analysis)
- [Error Analysis](#error-analysis)

## Introduction

This project demonstrates the use of a Variational Autoencoder (VAE) to learn a latent representation of MNIST digit images. The VAE model is trained to encode input images into a latent space and then decode them back to reconstruct the original images. The project includes steps for training the model, testing it, and performing various visualizations and analyses to understand the learned latent space.

## Model Architecture

The VAE consists of an encoder and a decoder:
- **Encoder:** Three convolutional layers to encode input images into a latent space.
- **Decoder:** Three transposed convolutional layers to reconstruct images from the latent space.
- **Reparameterization Trick:** Used to sample from the latent space during training.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/danielshort3/variational-autoencoder.git
    cd variational-autoencoder
    ```

2. Install the required packages (make sure you have `pip` installed):
    ```bash
    pip install torch torchvision tqdm matplotlib scikit-learn
    ```

## Usage

1. Run the code from beginning to end to train the model using `Variational_Autoencoder.ipynb` notebook.
2. After training, the notebook includes various sections for:
   - Visualizing the original and reconstructed images.
   - Exploring the latent space with dimensionality reduction techniques (PCA, t-SNE, UMAP).
   - Performing clustering analysis (HDBSCAN, DBSCAN).
   - Generating new images by sampling the latent space.
   - Analyzing reconstruction errors.

## Visualizations

The notebook provides visualizations such as:
- Original and reconstructed images.
- Latent space representations with PCA, t-SNE, and UMAP.
- Generated images by sampling from the latent space.
- UMAP plot with annotated outliers and cluster centroids.

## Data Analysis

The project includes analyses like:
- Clustering of latent space representations using HDBSCAN and DBSCAN.
- Calculation and visualization of cluster centroids.
- Outlier detection and nearest cluster calculation.
- Visualization of outliers and their reconstruction.

## Error Analysis

Error analysis includes:
- Displaying images with the highest and lowest reconstruction errors.
- Randomly selecting and displaying images with their reconstruction errors.
- Calculating and displaying average reconstruction error per digit.

For detailed explanations and visualizations, refer to the `Variational_Autoencoder.ipynb` notebook.
