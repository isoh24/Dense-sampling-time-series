# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 06:46:14 2021

@author: isor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
import random
import pickle

random.seed(0)

data_id=4 # 1: coindesk, 2: cdc ili, 3: power consumption, 4: wind speed

if data_id==1:    
    f=open("BTC_USD_2018-01-01_2020-12-31-CoinDesk.csv","r")
    coindesk_data=pd.read_csv(f,header=0)
    seq=coindesk_data[['Closing Price (USD)','24h Open (USD)','24h High (USD)','24h Low (USD)']].to_numpy()
    seq=coindesk_data[['Closing Price (USD)']].to_numpy() 
elif data_id==2:
    # CDC(Center for Disease Control and Prevention)의 ILI(influenza-like illness) data
    f=open("ILINet.csv","r")
    ili_data=pd.read_csv(f,header=1)
    seq=ili_data[['% WEIGHTED ILI']].to_numpy() 
elif data_id==3:
    f=open("household_power_consumption.txt","r")
    power_data=pd.read_csv(f,header=0,sep=';',na_values='?',keep_default_na=False)
    #seq=power_data[['Global_active_power', 'Global_reactive_power','Voltage', 'Global_intensity']].to_numpy()
    seq=power_data[['Global_active_power', 'Voltage', 'Global_intensity']].to_numpy()
    seq=power_data[['Global_intensity']].to_numpy()
    seq=seq[-10000:]
elif data_id==4:
    # Turbine data (with wind direction and wind speed)
    f=open("Turbine_Data.csv","r")
    wind_data=pd.read_csv(f,header=0)
    seq=wind_data[['WindSpeed']]
    seq=seq[-10000:].interpolate()
    seq=seq.to_numpy()
  
n_epochs=2000
batch_siz=32
n_trials=2
mode=1 # 1: experiment1(single horizon), 2: experiment2(multi horizons), 3: experiment3(effect of length of time series T)
   
def seq2dataset(seq,window,horizon):
    X=[]; Y=[]
    for i in range(window[0]-1,len(seq)-horizon):
        for k in range(len(window)):
            x=seq[i-window[k]+1:i+1]
            y=(seq[i+horizon])
            X.append(x); Y.append(y)
    return np.array(X), np.array(Y)

if mode==1: # sinle-step mode
    perf1,perf2,perf3,perf4=[],[],[],[]
    
    etime1,etime2=[],[]
    for w in range(18,21):    
        for s in range(10):
            frame=[w,w] 
            h=5 
            
            X,Y=[],[]
            for i in range(frame[0],frame[1]+1):
                x,y=seq2dataset(seq,[i],h)
                X.append(x)
                Y.append(y)
            
            x_train,y_train=[],[]
            x_test,y_test=[],[]
            
            split=int(len(seq)*0.3) # ratio for test set
            
            for i in range(len(X)):
                x_test.append(X[i][-split:])
                y_test.append(Y[i][-split:])
                x_train.append(X[i][0:-split])
                y_train.append(Y[i][0:-split])
            
            model = Sequential()
            model.add(Conv1D(32,3,activation='relu',padding='same',input_shape=(w,1)))
            model.add(Flatten())
            model.add(Dense(1))
            model.compile(loss='mae',optimizer='adam',metrics=['mae'])
            
            for i in range(n_epochs):
                for j in range(len(x_train)):              
                    model.train_on_batch(x_train[j],y_train[j])
                    
                pred=[]
                for j in range(len(x_train)):    
                    pred.append(model.predict(x_test[j]))
            
                #print(pred[0].shape);input('a')
                ave=np.mean(pred,axis=0)
                for k in range(len(pred)): 
                    pred[k]=ave
                        
                mape=np.mean(np.mean(abs(np.array(y_test)-pred)/abs(np.array(y_test)),axis=0),axis=0)
            
            perf1.append(mape)
            
            frame=[max(w-2,1),w] 
            h=5 
            
            X,Y=[],[]
            for i in range(frame[0],frame[1]+1):
                x,y=seq2dataset(seq,[i],h)
                X.append(x)
                Y.append(y)
            
            x_train,y_train=[],[]
            x_test,y_test=[],[]
            
            split=int(len(seq)*0.3) # ratio for test set
            
            for i in range(len(X)):
                x_test.append(X[i][-split:])
                y_test.append(Y[i][-split:])
                x_train.append(X[i][0:-split])
                y_train.append(Y[i][0:-split])
            
            pred=[]
            for j in range(len(x_train)):     
                model = Sequential()
                model.add(Conv1D(32,3,activation='relu',padding='same',input_shape=(x_train[j].shape[1],1)))
                model.add(Flatten())
                model.add(Dense(1))
                model.compile(loss='mae',optimizer='adam',metrics=['mae'])  
                
                for i in range(n_epochs):                     
                    model.train_on_batch(x_train[j],y_train[j])
                              
                #pred=[]
                pred.append(model.predict(x_test[j]))
                            
            ave=np.mean(pred,axis=0)
            for k in range(len(pred)):
                pred[k]=ave

            mape=np.mean(np.mean(abs(np.array(y_test)-pred)/abs(np.array(y_test)),axis=0),axis=0)
            
            perf2.append(mape)
            
