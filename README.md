# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries
2. Upload and read the dataset
3. Check for any null values using the isnull() function
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy
5. Find the accuracy of the model and predict the required values by importing the
6. Required module from sklearn

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: kavisree.s
RegisterNumber: 212222047001 
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt
data = pd.read_csv("/content/Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
data["Position"].value_counts()
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x = data[["Position","Level"]]
y=data["Salary"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

# Make predictions on the test set
y_pred = dt.predict(x_test)
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20,8))

plot_tree(dt,feature_names=x.columns, filled=True)
plt.show()
```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
<img width="394" alt="image" src="https://github.com/kavisree86/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145759687/5a5e9752-5fe1-4e2c-ba39-782d350bbf8e">

<img width="259" alt="image" src="https://github.com/kavisree86/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145759687/dabda670-1fce-4a90-b72a-6dabea9985a9">

<img width="274" alt="image" src="https://github.com/kavisree86/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145759687/8daeca92-f67a-4efc-aab8-91c1f6d1828b">

<img width="227" alt="image" src="https://github.com/kavisree86/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145759687/4d5789cc-c7a4-4f0b-b2ec-bd9db5b4cb9e">

<img width="605" alt="image" src="https://github.com/kavisree86/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145759687/8a2f269a-274a-4e12-bfb3-4cd2b186dfea">

<img width="512" alt="image" src="https://github.com/kavisree86/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145759687/369ed972-49a4-4396-a7c1-6cd11dc97274">




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
