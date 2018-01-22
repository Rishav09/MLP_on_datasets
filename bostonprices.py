#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:26:48 2018

@author: rishav
"""
import numpy
from pandas import read_csv
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

dataframes=read_csv("housing.csv",header=None,delim_whitespace=True)
dataset=dataframes.values
seed=7
numpy.random.seed(seed)
X=dataset[:,0:13]
Y=dataset[:,13]

def baseline_model():
    model=Sequential()
    model.add(Dense(13,input_dim=13,kernel_initializer='normal',activation='relu'))
    model.add(Dense(1,kernel_initializer='normal'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

estimator=KerasRegressor(build_fn=baseline_model,epochs=100,batch_size=5,verbose=0)
kfold=KFold(n_splits=10,random_state=seed)
results=cross_val_score(estimator,X,Y,cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(),results.std()))
