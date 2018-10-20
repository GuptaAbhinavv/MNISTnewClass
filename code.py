# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 18:55:19 2018

@author: abhinav
"""

import pandas as pd
import numpy as np


#Importing training and testing datasets
dataset = pd.read_csv("train.csv")
testset = pd.read_csv('test.csv')

#Selecting top 5000 entries for model learning for less complexity
X = dataset.iloc[:5000, 1:].values
Y = dataset.iloc[:5000, 0:1].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
X = sc1.fit_transform(X)
test = sc1.fit_transform(testset)

#One hot encoding the labels
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
Y = onehotencoder.fit_transform(Y).toarray()


from keras.models import Sequential
from keras.layers import Dense

#Building a simple ANN with one hidden layer and SIGMOID activation function for final layer
classifier = Sequential()
classifier.add(Dense(300, activation = 'relu', input_dim = 784 ))
classifier.add(Dense(10, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(X, Y, batch_size = 10, epochs = 5)

#Array with conditional probabilities predicted for each class
y_pred = classifier.predict(test)



#importing dataset containing random images
not_numbers = pd.read_csv("random_images")
not_number = not_numbers.values

not_number = sc1.fit_transform(not_number)

not_number_pred = classifier.predict(not_number)

#Displaying the values of maximum conditional probabilities for each entry
np.amax(not_number_pred, axis=1)

#Function for adding "NaN" to the class values with probabilities less than threshold 0.5
def add_nota(predictions):
    x = len(predictions)
    prob_max = np.amax(predictions, axis=1).reshape(x, 1)
    num_pred = np.argmax(predictions, axis =1).reshape(x, 1)

    arr = np.append(prob_max, num_pred, axis=1)

    for i in range(0, x):
        if arr[i][0]<0.5:
            arr[i][1] = "NaN"
            
    return arr

#Adding "NaN" to the predictions of dataset with random pictures
classes_not_number = add_nota(not_number_pred)
classes_not_number
#No of entries labeled as NaN is high(15 out of 16) which proves efficiency of the model.
#However, threshold can be varied for optimal results in case of larger dataset.
