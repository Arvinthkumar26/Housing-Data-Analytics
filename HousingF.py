#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:18:15 2019

@author: Arvinthkumar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# reading the csv file 
housing = pd.read_csv('housing.csv') 

housing.info()
    
housing.isnull().sum()

housing['total_bedrooms']= housing['total_bedrooms'].fillna(np.mean(housing['total_bedrooms']))

housing['avg_rooms'] = housing['total_rooms']/housing['households']
housing['avg_bedrooms'] = housing['total_bedrooms']/housing['households']
housing['pop_household'] = housing['population']/housing['households']

housing.corr()

housing.hist(bins=50, figsize=(20,20))
plt.show()

housing = pd.get_dummies(housing)

housing = housing.drop(["ocean_proximity_NEAR OCEAN","total_rooms","total_bedrooms",
                        "population"],axis=1)

X = housing.loc[:,housing.columns!='median_house_value'].values
y = housing.loc[:,housing.columns=='median_house_value'].values
print(X)
print(y)

#Spliting into train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression()
LogReg.fit(X_train,y_train)
y_pred = LogReg.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Accuracy
from sklearn.metrics import accuracy_score 
score = accuracy_score(y_test,y_pred)
print('Accuracy :',score)

# RandomForest
from sklearn.ensemble import RandomForestRegressor
RForest = RandomForestRegressor()
RForest.fit(X_train,y_train)
print(RForest.score(X_train,y_train))
pred=RForest.predict(X_test)

from sklearn import metrics
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
RForest.score(X_test, y_test)