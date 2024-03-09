#Implementation
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from customlogreg import customlogisticregression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# import dataset
df = pd.read_csv('D:/Advance Machine Learning/heart.csv')
df.head(6)

# drop missing values
df.dropna()

# Define X and y
X = df.drop('target',axis=1)
y = df['target']

# split data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# train model with modification parameter
model = customlogisticregression()
model.fit(X_train,y_train)

# predict model
y_pred = model.predict(X_test)

# show accuracy
print(accuracy_score(y_test,y_pred))

## Hyperparameter TUning
param_grid = {
    'learning_rate': [0.001, 0.01],
    'iterations': [1000, 2000 ]
}

# GridSearchCV for hyperparameter tuning
grid_cv = GridSearchCV(model, param_grid=param_grid, cv=3)
grid_cv.fit(X_train, y_train)

print("Best hyperparameters (GridSearchCV):", grid_cv.best_params_)
print("Best score (GridSearchCV):", grid_cv.best_score_)

# Predict using the best model from GridSearchCV
best_lr_model = grid_cv.best_estimator_
predictions = best_lr_model.predict(X_test)

# Evaluate the model
accuracy = np.sum(predictions == y_test) / len(y_test)
print("Accuracy:", accuracy)