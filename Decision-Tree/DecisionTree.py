import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

# https://www.kaggle.com/datasets/ziya07/employee-attrition-prediction-dataset
# Information on employees and the result of their personal turnover at a company.
df = pd.read_csv("employee_attrition_dataset.csv")

df = df[['Age', 'Monthly_Income', 'Hourly_Rate',
        'Years_at_Company', 'Years_in_Current_Role' ,'Years_Since_Last_Promotion',
        'Job_Satisfaction', 'Performance_Rating', 'Average_Hours_Worked_Per_Week',
         'Attrition']]

cat_columns = df.select_dtypes("object").columns
df[cat_columns] = df[cat_columns].astype("category")

cat_dict = {cat_columns[i]: {j: df[cat_columns[i]].cat.categories[j] for j in
                             range(len(df[cat_columns[i]].cat.categories))} for i in range(len(cat_columns))}

X = df[['Age', 'Monthly_Income', 'Hourly_Rate',
        'Years_at_Company', 'Years_in_Current_Role' ,'Years_Since_Last_Promotion',
        'Job_Satisfaction', 'Performance_Rating', 'Average_Hours_Worked_Per_Week',]]
y = df['Attrition']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Balance the data due to the different size of the classification sets
clf = DecisionTreeClassifier(class_weight='balanced')
clf.fit(X_train, y_train)

clf = DecisionTreeClassifier(class_weight='balanced')
clf.fit(X_train, y_train)
parameters = {"max_depth": range(2,20)} #Numbers 2 - 19

# Perform Grid Search for Max Depth
grid_search = GridSearchCV(clf, param_grid=parameters, cv=5)
grid_search.fit(X_train, y_train)
max_depth = grid_search.best_params_["max_depth"]
clf = DecisionTreeClassifier(max_depth=max_depth, class_weight='balanced')

clf.fit(X_train, y_train)

score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[['param_max_depth', 'mean_test_score', 'rank_test_score']])

# Print results of grid search and run
print(f"max_depth: {max_depth}")
print(f"Train Score: {clf.score(X_train, y_train):.3f}")
print(f"Test Score: {clf.score(X_test, y_test):.3f}")

# display a tree of height 3, appropriately labeled
clf = DecisionTreeClassifier(class_weight='balanced', max_depth=3)
clf.fit(X_train, y_train)
plt.figure(figsize=(20, 20))
plot_tree(clf, filled=True, feature_names=df.columns[:-1], class_names=['No','Yes'])
plt.show()

# Use class weighting? why or why not?
#I used class weighting because 19% of the observations were employees that left.
#Meanwhile, 81% were current employees.