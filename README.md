# Flower-Recognition-and-Detection-with-PyTorch

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methods with PyTorch](#methods-with-pytorch)
- [Visualization and Preprocessing](#visualization-and-preprocessing)
- [Transfer Learning with timm](#transfer-learning-with-timm)
  - [Model Architecture](#transfer-learning-model-architecture)
  - [Results of Transfer Learning](#results-of-transfer-learning)
- [Object Detection with YOLO](#object-detection-with-yolo)
  - [Implementation](#implementation)
  - [Results of Object Detection](#results-of-object-detection)
- [Key Insights](#key-insights)
- [How to Run](#how-to-run)
- [EXTRA: HTML visualization with DataPane](#EXTRA-html-visualization-with-datapane)


## 🌟 Project Overview

This repository is the seventh project of the master's degree in AI Engineering with [Profession AI](https://profession.ai), all the credits for the requests and idea go to this team. 

GreenTech Solutions Ltd., a pioneer in the agritech industry, is faced with the critical need to implement an advanced automatic flower recognition system within its daily operations. GreenTech Solutions Ltd. has started a strategic initiative to develop a prototype AI-based Computer Vision model for automatic flower recognition.

Project Benefits:
- Increased Productivity
- Improved Quality
- Technological Innovation

The main goal is to develop a highly robust model, capable of automatically classifying flowers with the best possible F1-score (macro) on the test dataset. I need to use techniques such as:
- Augmentation Strategies: implement different augmentation techniques to enrich the dataset, improving data variability and quality.
- Network Architectures and Transfer Learning: select and implement one or more neural network architectures wiht PyTorch's timm library, using transfer learning to exploit pre-trained models.
- Validation and Regularization: retraining with validation and regularization techniques to improve the model performance.
- Object detection: this part has to be done using YOLO models and supervsion library from Roboflow.

## 🍽️ Dataset

The dataset provided by ProfessionAI ([link](https://proai-datasets.s3.eu-west-3.amazonaws.com/progetto-finale-flowes.tar.gz)) includes two main categories of flowers:

- Daisy: 529 training images, 163 validation, 77 test.
- Dandelion: 746 training images, 201 validation, 105 test.

## 🛠️ Methods with PyTorch

This project leverages PyTorch to implement a robust and flexible system for training and evaluating CNN models. I've designed several custom classes to streamline the experimental process, you can find them [here](src/models.py)

### Experiment Class

The `Experiment` class serves as the backbone of the training pipeline. It manages:

- Logging of training progress
- Saving and loading of model weights
- Visualization of training history
- Exporting of results

Key features:
- Automatic creation of directory structure for each experiment
- CSV logging of training and validation metrics
- Plotting of training history
- JSON export of final results

### Callback System

I've implemented a callback system inspired by Keras, allowing for flexible control of the training process:

1. **EarlyStopping**: Prevents overfitting by stopping training when a monitored metric has stopped improving.
2. **ModelCheckpoint**: Saves the best model based on a specified metric.
3. **ReduceLROnPlateau**: Reduces learning rate when a metric has stopped improving.

### Model Architecture

This project leverages PyTorch along with the `timm` (PyTorch Image Models) library as the primary framework for model implementation. The choice of `timm` over traditional `torchvision` models was driven by several key advantages:

- Access to a wider range of state-of-the-art models
- Better documentation and community support
- Consistent API across different model architectures
- Regular updates with the latest architectures
- Pre-trained weights availability

For the classification task, I experimented with three carefully selected models:
1. `efficientnet_b5.sw_in12k_ft_in1k`: A top-performing model known for its efficiency-accuracy trade-off
2. `convnext_base`: A more recent architecture that shows strong performance
3. `resnet50`: A classic architecture serving as a reliable baseline

Each model was tested in two configurations:
- Direct fine-tuning with modified classification head
- Custom classifier addition with trainable layers

### Training and Evaluation Functions

The `train_model` function encapsulates the entire training loop, including:

- Epoch-wise training and validation
- Logging of metrics
- Execution of callbacks
- Resuming training from checkpoints

The `validate` and `get_predictions` functions provide easy-to-use interfaces for model evaluation and inference.


## 🖼️ Visualization and Preprocessing

I conducted a comprehensive Exploratory Data Analysis to understand this dataset better. The full EDA can be found in the [project notebook](project_notebook.ipynb), but here are some key insights:

- Class Distribution: The dataset shows a slight imbalance between classes:
  - Daisy: 529 training images (41.5%), 163 validation images (44.8%), 77 test images (42.3%)
  - Dandelion: 746 training images (58.5%), 201 validation images (55.2%), 105 test images (57.7%)
While there is an imbalance, the ratio remains consistent across splits and is not severe enough to significantly impact binary classification.

- Image Dimensions: All images are consistently sized at 512x512 pixels.

- Color Distribution: Analysis of a sample of images revealed a tendency towards green tones, with less representation in the blue spectrum, as expected in a flowers dataset.

For a more detailed visual exploration of the dataset, including class distribution plots, image dimension analysis, and color histograms, please refer to the EDA section in the project notebook. Functions code is [here](src/visualization.py)

### Preprocessing and Data Augmentation

Preprocessing pipeline was implemented using the `Albumentations` library, chosen for its efficiency and wide range of image augmentation techniques. The pipeline includes:

1. Resizing: All images are resized to 224x224 pixels to standardize input for these models.
2. Normalization: Images are normalized using mean and standard deviation values of (0.5, 0.5, 0.5) for each channel.

For data augmentation, I implemented a comprehensive approach that includes:

1. Basic geometric transformations:
   - `A.RandomRotate90()`
   - `A.Flip()`
   - `A.Transpose()`

2. Image enhancement transformations (with 0.2 probability):
   - `A.Sharpen(alpha=(0.2, 0.3), lightness=(0.5, 0.7))`
   - `A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)`

These augmentation techniques were carefully selected to preserve the natural appearance of flowers while increasing dataset variability.

You can find [here](src/preprocessing.py) the class Transforms and methods I've used for this project step.


## 🚀 Transfer Learning with timm

### Model Architecture

### Results of Transfer Learning

## Object Detection with YOLO

### Implementation

### Results of Object Detection

## Key Insights

## How to Run

## EXTRA: HTML visualization with DataPane
