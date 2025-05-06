# Support Vector Classifier for Raisin Classification

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

<img src="https://github.com/user-attachments/assets/6f7a3d44-56f0-4a03-886c-de44f68bb909" alt="Early Blight" width="300"/>
<img src="https://github.com/user-attachments/assets/427c7c4b-2b8d-4243-9cae-7a05ae0200e4" alt="Late Blight" width="300"/>
<img src="https://github.com/user-attachments/assets/4f286249-99f5-40f2-9818-21b7db943bf8" alt="Healthy" width="300"/>

## Project Run Through

### Initialize Data

Image data is taken in and declared as train or test based on their main directory. Then, all images are resized to be 96x96 and converted to tensors.

### Model and Training Inputs

The model has two convolutional layers with BatchNorm2d and ReLU. It trains using the Adam optimizer and a CrossEntropyLoss function.

Training the model on 100 epochs with a 0.0001 learning rate and 32 batch size results in the accuracy getting stuck around .87. However, lowering the batch size to 16 and increasing the learning rate results in the accuracy increasing to around .89.

Training the model with 50 epochs results in the model performing with an accuracy of .8638. However, looking at the confusion matrix below. The difference between healthy and 'any blight' is performing at .88. 

<img src="https://github.com/user-attachments/assets/583533c7-52cf-45c8-a955-6a7cbe9326f0" alt="Confusion Matrix - 50 Epochs" width="500"/>

However, running on 100 epochs results in an increased overall accuracy of .8812. With an addition to this, the difference between healthy and 'any blight' has increased as well.

Thus, the following results are based on a <b>batch size of 16</b> with a <b>learning rate of 0.001</b> running on <b>100 epochs</b>.

### Results

The final accuracy on the test set is 0.8812. The confusion matrix below details these results further. Although this is a small jump from 50 epochs, as mentioned before, the performance of healthy vs 'any blight' has risen to .94.

<img src="https://github.com/user-attachments/assets/501ff45d-d888-4b25-85d9-fc11d5629fdc" alt="Confusion Matrix - 100 Epochs" width="500"/>

## Reflection

Although the final accuracy of the model is .8812. Detecting early vs late blight is not as important as detecting healthy vs any blight. As seen in the confusion matrix above, the accurate prediction of a healthy leaf is greater than either of the other categories. So, even though the accuracy is not consistently reaching 90%, the model is still effective in detecting disease.
