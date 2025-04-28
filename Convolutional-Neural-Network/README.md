# Convolutional Neural Network for Tomato Leaf Disease Classification

## Overview
### Machine Learning Technique

A convolutional neural network is a supervised deep learning algorithm. As an expansion of artificial neural networks, they are used for image recognition and classification tasks. 

### Program's Purpose

This program is used to identify early and late stage blight in photos of tomato leaves. Using a dataset that contains lab and wild photos of healthy, early blight infected, and late blight infected tomato leaves, the network classifies each type in order to work towards helping amateur tomato gardeners identify blight with their own photos of their plants. 

## Data

This program uses images from the [Tomato Leaves Dataset](https://www.kaggle.com/datasets/ashishmotwani/tomato/data). I pulled images from three directories for each train and test main directory. The breakdown of the number of files for each directory is detailed in the table below.

| Directory Name | Number of Files | Main Directory |
|:--------------:|:---------------:|----------------|
| Early_blight   | 2455            | train          |
| Late_blight    | 3113            | train          |
| healthy        | 3051            | train          |
| Early_blight   | 643             | test           |
| Late_blight    | 792             | test           |
| healthy        | 805             | test           |

Here are example images from the dataset for each category: Early blight, Late blight, and healthy.

<img src="https://github.com/user-attachments/assets/6f7a3d44-56f0-4a03-886c-de44f68bb909" alt="Early Blight" width="500"/>
<img src="https://github.com/user-attachments/assets/427c7c4b-2b8d-4243-9cae-7a05ae0200e4" alt="Late Blight" width="500"/>
<img src="https://github.com/user-attachments/assets/4f286249-99f5-40f2-9818-21b7db943bf8" alt="Healthy" width="500"/>

## Project Run Through

### Initialize Data

Image data is taken in and declared as train or test based on their main directory. Then, all images are resized to be 96x96 and converted to tensors.

### Model and Training Inputs

The model has two convolutional layers with BatchNorm2d and ReLU. It trains using the Adam optimizer and a CrossEntropyLoss function.

Training the model on 100 epochs with a 0.0001 learning rate and 32 batch size results in the accuracy getting stuck around .87. However, lowering the batch size to 16 and increasing the learning rate results in the accuracy increasing to around .89.

Training the model with 50 epochs results in the model performing with an accuracy around .87. However, looking at the confusion matrix below. The difference between healthy and 'any blight' is performing at .*. 

<img src="https://github.com/user-attachments/assets/6f7a3d44-56f0-4a03-886c-de44f68bb909" alt="Confusion Matrix - 50 Epochs" width="500"/>

However, running on 100 epochs results in an increased overall accuracy of around *. Although this is a small jump from 50 epochs, the performance of healthy vs 'any blight' has risen to *. 

Thus, the following results are based on a <b>batch size of 16</b> with a <b>learning rate of 0.001</b> running on <b>100 epochs</b>.

### Results

The final accuracy on the test set is *. The confusion matrix below details these results further. 

![Confusion_Matrix]()

## Reflection

Lowering the learning rate resulted in the final accuracy getting stuck around .87 while keeping a batch size of 32 and running 100 epochs. Decreasing the epochs appeared to help as eventually the accuracy began decreasing. Decreasing the batch size seems to have helped as well. 

Although the final accuracy of the model is *. Detecting early vs late blight is not as important as detecting healthy vs any blight. As seen in the confusion matrix above, the accurate prediction of a healthy leaf is greater than either of the other categories. So, even though the accuracy is not consistently reaching 90%, the model is still effective in detecting disease.
