# Deep Learning in Data Science — Final Project

This repository contains the final project for the course **DD2424 - Deep Learning in Data Science** at **KTH Royal Institute of Technology**.

---

## Project Overview

The project implements a deep learning model based on the **AlexNet** architecture using **TensorFlow** to perform image classification on the **Oxford 17 Category Flower Dataset** and the **Oxford 102 Category Flower Dataset**.

The goal is to demonstrate convolutional neural networks' ability to classify fine-grained categories of flowers with high accuracy by leveraging transfer learning and data augmentation techniques.

---

## Datasets

- [Oxford 17 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
- [Oxford 102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

Both datasets consist of high-quality images with detailed annotations for multiple flower categories.

---

## Architecture

- **Model:** AlexNet CNN architecture
- **Framework:** TensorFlow 2.x
- **Key features:**
  - Multiple convolutional layers with ReLU activations
  - Max-pooling layers for downsampling
  - Fully connected dense layers for classification
  - Dropout regularization to reduce overfitting

---

## Usage

### Installation

Make sure you have Python 3.7+ and TensorFlow installed. It is recommended to use a virtual environment.


```bash
pip install -q -U tensorflow_hub
pip install -q tfds-nightly tensorflow matplotlib
pip install tflearn
```
---

## Training & Evaluation
To train the AlexNet model on the datasets, see the code under the Project folder: 
- [17 category dataset](./Project/17_category_flowers_AlexNet.ipynb)
- [102 category dataset](./Project/102_category_flowers_AlexNet.ipynb)


