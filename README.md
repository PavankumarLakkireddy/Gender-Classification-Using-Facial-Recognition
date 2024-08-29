# Gender Classification using VGG16

This project implements a gender classification model using a fine-tuned VGG16 neural network architecture. The model is trained to classify images as either "Male" or "Female".

## Table of Contents
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- scikit-learn
- numpy
- PIL

## Dataset

The dataset is structured as follows:
- `/kaggle/input/gender-classification/Gender Dataset/train`
- `/kaggle/input/gender-classification/Gender Dataset/valid`
- `/kaggle/input/gender-classification/Gender Dataset/test`

Each directory contains subdirectories for "Female" and "Male" images.

## Model Architecture

The model uses a pre-trained VGG16 network with the following modifications:
- The final fully connected layer is replaced to output 2 classes (Male/Female)
- All layers except the classifier are frozen

## Training

- Optimizer: Adam
- Learning Rate: 0.0001
- Loss Function: Cross Entropy Loss
- Epochs: 60
- Batch Size: 64 (train), 4 (validation/test)

Data augmentation techniques applied:
- Random horizontal flip
- Random vertical flip
- Random grayscale conversion
- Gaussian blur

## Results

After 60 epochs:
- Validation Accuracy: 90.70%
- Validation Loss: 0.0614

Test Results:
- Testing Accuracy: 95.54%
- Testing Loss: 0.0419

Classification Report (Test Data):
```
              precision    recall  f1-score   support

      Female       1.00      0.91      0.95        55
        Male       0.92      1.00      0.96        57

    accuracy                           0.96       112
   macro avg       0.96      0.95      0.96       112
weighted avg       0.96      0.96      0.96       112
```

## Usage

1. Prepare your dataset in the structure mentioned above.
2. Run the script to train the model:
   ```
   python train_gender_classifier.py
   ```
3. The script will output training progress, validation results, and finally test results.
4. Plots for training/validation loss and accuracy will be displayed.

Feel free to modify hyperparameters or model architecture to experiment with different configurations.
