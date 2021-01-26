# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 00:20:12 2020

@author: Acer
"""

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Preparing Data

# Importing the dataset
dataset = pd.read_csv('Customer_Dataset.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
dataset

X.shape


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(sparse = False)
one = onehotencoder.fit_transform(X[:,1].reshape(-1,1))#.toarray()
# X = X[:, 1:]

one = one[:,0:2]

X = np.delete(arr=X,obj=1,axis=1)

X_1 = X[:,0].reshape((-1,1))

X_2 = one

X_3 = X[:,1:]

X_half = np.append(X_1,X_2,1)

X_fin = np.append(X_half,X_3,1)
X_fin.shape

X = X_fin.copy()
X.shape

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train[0]

X_test.shape


# Building Neural Network
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input Layer and the First hidden layer
classifier.add(Dense(units = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p = 0.1))

# Adding the secong hidden layer
classifier.add(Dense(units = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, init = 'uniform', activation = 'sigmoid'))

classifier.summary()

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)



#  Predicting the test set results
y_predict = classifier.predict(X_test)
y_pred = (y_predict > 0.5)
y_pred[:10]



# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy rate of predictons
(2317 + 204)/3000

# Saving the Model
classifier.save("Trained model(11).model")
#  How to load a trained model
# keras.models.load_model("address")



#  Evaluating and Improving The ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)

# mean = accuracies.mean()
# variance = accuracies.std()


# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = { 'batch_size' : [25, 32],
               'epochs'     : [100, 200],
               'optimizer'  : ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           cv = 5)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

