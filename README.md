# Fashion-Mnist-Cnn-model-optimized-using-keras-tuner

This repository contains a Convolutional Neural Network (CNN) model for classifying images from the Fashion-MNIST dataset. The model is optimized using Keras Tuner to find the best hyperparameters for improved accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Results](#results)

## Introduction

Fashion-MNIST is a dataset of Zalando's article images, consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image associated with a label from 10 classes. This project aims to build a CNN model for classifying these images and optimize the model using Keras Tuner.

## Dataset

The Fashion-MNIST dataset includes the following classes:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

The dataset can be loaded directly using TensorFlow:

```python
from tensorflow.keras.datasets import fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

## Model Architecture

The CNN model consists of several convolutional layers, followed by flatten and then dense layers. The architecture is defined using Keras and optimized using Keras Tuner.

## Hyperparameter Optimization

Keras Tuner is used to search for the best hyperparameters for the CNN model. The following hyperparameters are tuned:
- Number of filters in each convolutional layer
- Kernel size
- Learning rate

## Results

The best model found by Keras Tuner achieves a high accuracy on the test set. Detailed results and the best hyperparameters are documented in the notebook.
