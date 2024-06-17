import pandas  as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
t=pd.read_csv('titanic.csv')
print(t)
t.head()
print(t.head())
t.isnull().sum()
import seaborn as sns
import matplotlib.pyplot as plt
print(sns.heatmap(t.isnull()))
#visualization
sns.countplot(x='Survived',data=t)
#GENDER
sns.countplot(x='Survived',hue='Sex',data=t)
#GENDER
sns.countplot(x='Survived',hue='Age',data=t)
t.isnull().sum()
t.isnull().sum()
t.dropna(subset=['Age', 'Cabin', 'Embarked'], inplace=True)
t
print(t.isnull().sum())
t.describe()
t.info()
#drop certain variables
t.drop(['Ticket','Cabin','Name','PassengerId'],axis=1,inplace=True)
sns.countplot(t['Embarked'])
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Country'.
t['Sex']= label_encoder.fit_transform(t['Sex'])
print(t['Sex'])
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column '
t['Embarked']= label_encoder.fit_transform(t['Embarked'])
t.corr()
sns.heatmap(t.corr(),annot=True)
x=t[['Survived','Pclass','Sex','Age','Parch','Fare','Embarked']]
y=t['Survived']
from sklearn.tree import DecisionTreeClassifier  # or DecisionTreeRegressor for regression tasks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # or other appropriate metrics

# Assuming X is your feature matrix and y is your target variable
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=42)

# Initialize the decision tree classifier
model = DecisionTreeClassifier()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict the target variable on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
import pickle
import pickle

# Assuming 'model' is the object you want to save
with open('modeltitanic.pkl', 'wb') as f:
    pickle.dump(model, f)
import pickle

with open('modeltitanic.pkl', 'rb') as f:
    z = pickle.load(f)

# 'z' now contains the loaded model


with open('modeltitanic.pkl','rb') as f:
    z = pickle.load(f)
feature_matrix = [[1,	1	,0	,38.0	,0	,71.2833	,0]]

# Predict the target variable on the feature matrix
y_pred = z.predict(feature_matrix)
print(y_pred)