# Implementation of Simple Linear Regression Model for Predicting the Marks Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```txt
1. Use the standard libraries in python.
2. Set variables for assigning data set values.
3. Import Linear Regression from the sklearn.
4. Assign the points for representing the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtain the LinearRegression for the given data.
```

## Program:
```txt

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Krupa Varsha P
RegisterNumber: 212220220022 

```

```python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('/content/student_scores.csv')
print("data.head():")
data.head()
```

```python3
print("data.tail():")
data.tail()
```

```python3
x=data.iloc[:,:-1].values
x
```
```python3
y=data.iloc[:,1].values
y
```
```python3
print(x)
print(y)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0 )
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(y_pred)
y_pred
```
```python3
print(y_test)
y_test
```
```python3
#for train values
plt.scatter(x_train,y_train)
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.title("Hours Vs Score(Training set)")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()
```
```python3
#for test values
y_pred=regressor.predict(x_test)
plt.scatter(x_test,y_test)
plt.plot(x_test,regressor.predict(x_test),color='black')
plt.title("Hours Vs Score(Test set)")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()
```
```python3
import sklearn.metrics as metrics
mae = metrics.mean_absolute_error(x, y)
mse = metrics.mean_squared_error(x, y)
rmse = np.sqrt(mse)
print("MAE:",mae)
print("MSE:", mse)
print("RMSE:", rmse)
```
## Output:
#data.head()

![14191a45-9eed-45af-bae5-fae2a314c075](https://user-images.githubusercontent.com/100466625/230769245-312aa4b0-22eb-46a3-978b-9d3e1031b82f.jpg)



#data.tail()
![708d194f-a66a-4ae8-8a07-7fc8051a2b71](https://user-images.githubusercontent.com/100466625/230769254-da03bd82-74a3-4065-b59e-0819c561d68d.jpg)

#Array value of x
![d49dedf2-f08f-48df-8207-73b96bc242d7](https://user-images.githubusercontent.com/100466625/230769499-7f8ec4c5-af58-456e-8625-92542c2beecb.jpg)


#Array values of y
![2d41b5b3-4e8e-4989-a227-af76175669a8](https://user-images.githubusercontent.com/100466625/230769290-8851f619-7c30-4de5-9117-ce15a415e64f.jpg)


#values of Y predication
![8c1424d2-53f4-44cb-a187-f1e75140a431](https://user-images.githubusercontent.com/100466625/230769354-f35b80ef-a69b-43bc-9cca-b84b7e4bf078.jpg)

#Array value of Y Test
![ab38b568-cf5b-416c-9569-be2f764a6982](https://user-images.githubusercontent.com/100466625/230769315-2aefa798-880f-4942-b7a3-d6bb665b9140.jpg)

#Training Set Graph

![00bae286-4034-46a4-b6a1-3ee4a0a45b69](https://user-images.githubusercontent.com/100466625/230769372-3e5e9cbf-1e79-4803-8ca3-1de61c61d3dc.jpg)

#Test Set Graph

![c8748bb7-9fad-483e-9ec3-753a88ac1acc](https://user-images.githubusercontent.com/100466625/230769387-3fbf1829-8012-4ec5-837c-56040b52fe41.jpg)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
