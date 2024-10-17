# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## DATE:
## AIM:
To implement simple linear regression using sklearn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y by reading the dataset.
2. Split the data into training and test data.
3. Import the linear regression and fit the model with the training data.
4. Perform the prediction on the test data.
5. Display the slop and intercept values.
6. Plot the regression line using scatterplot.
7. Calculate the MSE.

## Program:
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: pranav k
RegisterNumber:  2305001026
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ex1.csv')
df.head()
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
from sklearn.model_selection import train_test_split
X = df['X']
y = df['Y']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
X_train_reshaped = X_train.values.reshape(-1, 1)
lr.fit(X_train_reshaped,Y_train)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train, lr.predict(X_train.values.reshape(-1, 1)), color='red')
m=lr.coef_
m
b=lr.intercept_
b
pred=lr.predict(X_test.values.reshape(-1, 1))
pred
X_test
Y_test
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_test, pred)
print(f"Mean Squared Error (MSE): {mse}")
```

## Output:
![Screenshot 2024-10-17 092736](https://github.com/user-attachments/assets/c7e9f568-cd40-46b4-bb92-ec654fb0f5e3)
![Screenshot 2024-10-17 092756](https://github.com/user-attachments/assets/d6e77182-f7db-40e6-876d-5228b33367aa)
![Screenshot 2024-10-17 092820](https://github.com/user-attachments/assets/f60955af-5fb9-44d5-94ff-a07d118f4bbe)
![Screenshot 2024-10-17 092849](https://github.com/user-attachments/assets/bff05d66-13bc-4f53-9feb-7b6f7956bac0)
![Screenshot 2024-10-17 092856](https://github.com/user-attachments/assets/7f062524-fb26-4d15-84b0-43a1c4c54dbb)



## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
