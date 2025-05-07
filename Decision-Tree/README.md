# Decision Tree for Employee Attrition Classification

## Overview
### Machine Learning Technique

Decision Tree Classification is a supervised learning technique used to classify data into known classes. It uses nodes and branches to create groupings of classes for classification. A large benefit of this technique is being able to easily follow the nodes and branches to reach a conclusion. 

### Program's Purpose

This program is used to classify employee attrition, or the turnover, of a company. It generates a decision tree which can be followed to determine if an employee is likely to be at the company still or if they have left. It finds the optimal depth of the tree to determine how to recieve the best results. However, trees can become very large quickly and thus a 3-depth tree has been used to explain the final results.

## Data

This program uses images from the [Employee Attrition Dataset](https://www.kaggle.com/datasets/ziya07/employee-attrition-prediction-dataset). Each employee has information regarding age, monthly income, hourly rate, years at the company, years in their current role, years since last promotion, job satisfaction, performance rating, and average hours worked per week. Each employee is then listed with an 'Attrition' label with either No or Yes. For this program, No means that the employee has not left the company and Yes means that the employee has left the company.

## Project Run Through

### Initialize and Find Data

Uses a test/train split with a test size of 30% to seperate the data. Then, uses balanced class weight due to the 19-81 split of the attrition result in the dataset. 

Then ran a grid search to find the optimal max depth parameter.

<img src="https://github.com/user-attachments/assets/c364dc6f-5e21-4b9a-9937-484e3e25bc9a" alt="Grid Search Results" width="500"/>

### Results

Below is a result of a tree with max depth 3. 

<img src="https://github.com/user-attachments/assets/7a05914c-4ebb-473a-ba0f-b9af172b031a" alt="Decision Tree"/>

In the above tree, the left branches represent the True result pathways while the right branches represent the False result pathways. Starting at the top, if an employee has worked at the company for less than or exactly 13.5 years then they will follow the left branch. Then, if that same employee has a monthly income of more than $5,671 a month they will follow the right path. And finally, if this employee works for more than 43.5 hours a week on average they will follow the right branch again. This 3-depth tree concludes that this employee will have left the company.

## Reflection

Predicting the behavior of people can be difficult and with limited data this model appears to overfit the train score with around 90% accuracy while the test score stays around 70% accurate. However, the general trends of the data still provide valuable insight in limiting turnover of employees.
