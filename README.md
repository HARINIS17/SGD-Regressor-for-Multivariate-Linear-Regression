# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import all required libraries.

2.Load the California housing dataset.

3.Convert it to a DataFrame and add the target column.

4.Split features and targets.

5.Do train and test split.

6.Normalize the data using StandardScaler.

7.Create an SGDRegressor with MultiOutputRegressor.

8.Train the model with training data.

9.Predict on test data and reverse the scaling.

10.Print mean squared error and some predictions.
## Program:
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.

Developed by: HARINI S 
RegisterNumber: 212224230083

```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset = fetch_california_housing()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['HousePrice'] = dataset.target
print(df.head())

X = df.drop(columns=['AveOccup', 'HousePrice'])
Y = df[['AveOccup', 'HousePrice']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)

Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

print("\nPredictions\n", Y_pred[:5])
```
## Output:
![Screenshot 2025-04-18 151249](https://github.com/user-attachments/assets/1308288a-ccac-46c3-94ab-cf3d5289f3a6)
![Screenshot 2025-04-18 151302](https://github.com/user-attachments/assets/5b64699b-4b04-42d1-9c81-88e6e77a30b6)
![Screenshot 2025-04-18 151321](https://github.com/user-attachments/assets/cb6a8c64-6904-47e0-a3c7-fb4335512a74)
![Screenshot 2025-04-18 151336](https://github.com/user-attachments/assets/46c40289-dc9d-45a2-bd27-42375c4ae03a)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
