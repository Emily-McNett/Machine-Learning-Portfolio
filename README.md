# Machine-Learning Portfolio
Programs Created in Machine Learning (MSCS 335) During the Spring 2025 Semester

Each folder contains the commented code, data (if file sizes allowed), and an individual README with a more thorough expantion of the program. Basic descriptions of each program's technique and purpose as well as the links to the program's folder can be found below.

## KMeans Clustering for Yarn Selection
### Machine Learning Technique

KMeans clustering is an unsupervised learning algorithm that partitions a dataset into 'k' clusters. One technique to find the optimal 'k' is to create a visualization called an elbow graph. In this graph, the 'elbow' refers to the point where an additional number of clusters no longer notably impacts the results. 

### Program's Purpose

This program is used to assist in the selection of yarn and yarn colors from a photo for use in various fiber arts. After uploading a photo and deciding on the number of dominant colors in an image, the dominant colors of the image, found through KMeans clustering, will be matched to the closest yarn color in the dataset. Users will then be able to compare the ‘true’ colors to the selected yarn.

### Folder Link
[KMeans-Clustering](https://github.com/Emily-McNett/Machine-Learning-Portfolio/tree/main/KMeans-Clustering)


## Convolutional Neural Network for Tomato Leaf Disease Classification
### Machine Learning Technique

A convolutional neural network is a supervised deep learning algorithm. As an expansion of artificial neural networks, they are used for image recognition and classification tasks. 

### Program's Purpose

This program is used to identify early and late stage blight in photos of tomato leaves. Using a dataset that contains lab and wild photos of healthy, early blight infected, and late blight infected tomato leaves, the network classifies each type in order to work towards helping amateur tomato gardeners identify blight with their own photos of their plants. 

### Folder Link
[Convolutional-Neural-Network](https://github.com/Emily-McNett/Machine-Learning-Portfolio/tree/main/Convolutional-Neural-Network)


## Support Vector Classifier for Raisin Classification
### Machine Learning Technique

A support vector classifier (SVC) is a method for classifying data into different groups. The method works to find the best seperation to maximize the distance between the 'support vectors', or closest points, within each class.

### Program's Purpose

This program is used to classify raisin types based on the given features of area, major and minor axis lengths, exxentricity, convex area, extent, and perimeter. Based on these features, the SVC will classify a variety of raisins as either Besni or Kecimen.

### Folder Link
[Support-Vector-Classifier](https://github.com/Emily-McNett/Machine-Learning-Portfolio/tree/main/Support-Vector-Classifier)

## Decision Tree to Detect Employee Attrition
### Machine Learning Technique

Decision Tree Classification is a supervised learning technique used to classify data into known classes. It uses nodes and branches to create groupings of classes for classification. A large benefit of this technique is being able to easily follow the nodes and branches to reach a conclusion. 

### Program's Purpose

This program is used to classify employee attrition, or the turnover, of a company. It generates a decision tree which can be followed to determine if an employee is likely to be at the company still or if they have left. It finds the optimal depth of the tree to determine how to recieve the best results. However, trees can become very large quickly and thus a 3-depth tree has been used to explain the final results.

### Folder Link
[Decision-Tree](https://github.com/Emily-McNett/Machine-Learning-Portfolio/tree/main/Decision-Tree)
