# Create a support vector classifier using radial basis functions for the kernel
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# https://www.kaggle.com/datasets/nimapourmoradi/raisin-binary-classification
# Using 450 of each Besni and Kecimen raisin varieties grown in Turkey
# The features are based on images of the raisins and are
# Area - The number of pixels of the raisin
# Major Axis Length - Longest line that can be drawn across
# Minor Axis Length - Shortest line that can be drawn across
# Eccentricity - ratio of the distance from any point to the focus to the directrix
# ConvexArea - The number of pixels of the smallest convex shell
#  Extent - Ratio of raisin pixels to total pixels in the bounding box
# Perimeter - Distance between the boundaries of the raisin and the box
df = pd.read_csv("Raisin_Dataset.csv")

cat_columns = df.select_dtypes("object").columns
df[cat_columns] = df[cat_columns].astype("category")

cat_dict = {cat_columns[i]: {j: df[cat_columns[i]].cat.categories[j] for j in
                             range(len(df[cat_columns[i]].cat.categories))} for i in range(len(cat_columns))}
print(cat_dict)

# select features and label
y = df.iloc[:, -1].copy().to_numpy()
X = df.iloc[:, :-1].copy().to_numpy()

#normalize data
X = (X - np.average(X, axis=0))/np.std(X, axis=0)
# y = (y - np.average(y))/np.std(y)

# Test-Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = SVC(kernel="rbf")
clf.fit(X_train, y_train)
#  use 2 to predict???
gam_basic = 1/(len(X[0]))
parameters = {"C": np.linspace(10, 100, num=10),
"gamma": np.linspace(gam_basic/10, gam_basic*10, num=10)}

grid_search = GridSearchCV(clf, param_grid=parameters, cv=5)
grid_search.fit(X_train, y_train)
C_best = grid_search.best_params_["C"]
gamma_best = grid_search.best_params_["gamma"]

clf = SVC(kernel="rbf", C=C_best, gamma=gamma_best)

clf.fit(X_train, y_train)

print(f"Best C: ", C_best)
print(f"Best Gamma: ", gamma_best)

score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[['param_C', 'param_gamma', 'mean_test_score', 'rank_test_score']])

print("Score: ", clf.score(X_test, y_test))

# Confusion Matrix
cm = confusion_matrix(y_test, clf.predict(X_test))
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()

# Predict Raisin Type -> Is score optimal
# The score is the accuracy of the % correct which I feel works well with this dataset
# It is a perfectly balanced dataset where both raisin types have 450 photos
# False negatives or false positives are not necessarily harmful as they are with
# other datasets.