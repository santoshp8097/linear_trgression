# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 19:59:26 2020

@author: Santosh
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#importing dataset
dataset=pd.read_csv('D:\\final data science\\datasets\\Salary_Data.csv')

print("no of indexes",str(len(dataset)))

print("salary",(dataset))

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

# taking out mising data(mostly not used)

from sklearn.preprocessing import Imputer 
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(x[:,0:2])
x[:,1:3]=imputer.transform(x[:,0:2])

#encoding catagorical data (mostly not used)

from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])

#encoding data into binary form(mostly not needed)

from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()

#encoding categorical data for y(no need)
from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
y=labelencoder_x.fit_transform(y)

#splitiing of data (no need)

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#feature scalling(no nedd)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)


#actual code for simple linear regrrssion

#fitting simple linear regeression to the training set 

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting the new test case result

y_pre=regressor.predict(x_test)
 #y_new_salary=regressor.predict(20)
 
 #visualizing a training set result
 
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs experiance(training set)')
plt.xlabel('years of experiance')
plt.ylabel('salary')
plt.show()
 
 #visualize the test case result
 plt.scatter(x_test,y_test,color='red')
 plt.plot(x_train,regressor.predict(x_train),color='blue')
 plt.title('salary vs experiance(training set)')
 plt.xlabel('years of experiance')
 plt.ylabel('salary')
 plt.show()
 
 y_new_salary=regressor.predict(20)