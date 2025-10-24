# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn 

 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KEERTHANA C
RegisterNumber:  212224220047
*/
```
```
import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()

data.info()
data.isnull().sum()
data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
```

## Output:

DATA HEAD:
<img width="1631" height="258" alt="image" src="https://github.com/user-attachments/assets/4c87bea6-b2cb-4a9e-bb16-82970e760844" />

DATASET INFO:
<img width="611" height="598" alt="image" src="https://github.com/user-attachments/assets/74bdf569-a9c9-495f-b897-b350c33b845b" />

DATASET TRANSFORMED HEAD:
<img width="1618" height="250" alt="image" src="https://github.com/user-attachments/assets/8b39999c-79a2-4113-ad35-5883a9fa09c5" />

ACCURACY:
<img width="400" height="50" alt="image" src="https://github.com/user-attachments/assets/579c5094-36de-48db-96ae-5501c38cf969" />

DATA PREDICTION:
<img width="1033" height="619" alt="image" src="https://github.com/user-attachments/assets/9ba48f3b-6374-4e9a-ae83-0488ac62e43e" />






## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
