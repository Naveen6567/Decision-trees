#Import below packages

import pandas as pd    # For data manipulation
import numpy as np     # For numerical calculation
import matplotlib.pyplot as plt # For data visualization
from matplotlib.colors import ListedColormap #For colors in data visualization
import seaborn as sns       # For advanced data visualization
from sklearn.preprocessing import normalize, LabelEncoder   # For standardizing the dataset
from sklearn.neighbors import KNeighborsClassifier    # For builiding KNN models
from sklearn.model_selection import train_test_split  # To split the data into training and test 
from sklearn.metrics import confusion_matrix          # To create confusion matrix
from sklearn.metrics import accuracy_score            # To measure the accuracy of model

#Import the dataset company_data from local machine which have 11 variables and 400 observations
company = pd.read_csv("F:/360DigiTmg/Module-14/Question-1/Company_Data.csv")
company1=company
#Check for any null values in dataset
company.isnull().sum()
#Find the column names in the dataset
company.columns

lb = LabelEncoder()
company["ShelveLoc"] = lb.fit_transform(company["ShelveLoc"])
company["Urban"] = lb.fit_transform(company["Urban"])
company["US"] = lb.fit_transform(company["US"])

#Convert sales data to categorical data, if sales value is less than 10 then 0 or 1
company.loc[company['Sales'] < 10, 'Sales'] = 0
company.loc[company['Sales'] > 10, 'Sales'] = 1

#Assign all inputs to predictors and output to Target
colnames = list(company.columns)
predictors = colnames[1:11]
target = colnames[0]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(company, test_size = 0.3)

#Import decision tree and build a decision tree using data
from sklearn.tree import DecisionTreeClassifier as DT

model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])

# Predict the decision tree on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# We got 84.16% test accuracy

# Predict the decision tree on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

# We got 100% train accuracy

