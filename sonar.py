#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:26:46 2018

@author: rishav
"""

import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
seed=7
numpy.random.seed(seed)

dataframe=read_csv("sonar.csv",header=None)
dataset=dataframe.values
X=dataset[:,0:60].astype(float)
Y=dataset[:,60]

encoder=LabelEncoder()
encoder.fit(Y)
encoded_Y=encoder.transform(Y)

def create_baseline():
    model=Sequential()
    model.add(Dense(60,input_dim=60,kernel_initializer='normal',activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

estimator=KerasClassifier(build_fn=create_baseline,epochs=100,batch_size=5,verbose=0)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(estimator,X,encoded_Y,cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100,results.std()*100))