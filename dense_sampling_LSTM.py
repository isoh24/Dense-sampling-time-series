# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 06:46:14 2021

@author: isor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import random
import pickle

random.seed(0)

data_id=1 # 1: coindesk, 2: cdc ili, 3: power consumption, 4: wind speed

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
   
# segmenting time series data into window
def seq2dataset(seq,window,horizon):
    X=[]; Y=[]
    for i in range(window[0]-1,len(seq)-horizon):
        for k in range(len(window)):
            x=seq[i-window[k]+1:i+1]
            y=(seq[i+horizon])
            X.append(x); Y.append(y)
    return np.array(X), np.array(Y)

if mode==1: # single-step mode
    perf1,perf2=[],[]
    
    etime1,etime2=[],[]
    for w in range(2,13):
        for s in range(10):
            frame=[w,w] # frame
            h=5 # horizon factor
            
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
            
            # LSTM model implementation
            model = Sequential()
            model.add(LSTM(units=128,activation='relu',input_shape=(None,1)))
            model.add(Dense(1))
            model.compile(loss='mae',optimizer='adam',metrics=['mae'])
            
            for i in range(n_epochs):
                for j in range(len(x_train)):            
                    model.train_on_batch(x_train[j],y_train[j])
                    
                pred=[]
                for j in range(len(x_train)):    
                    pred.append(model.predict(x_test[j]))
            
                ave=np.mean(pred,axis=0)
                for k in range(len(pred)): # replacing pred with average
                    pred[k]=ave
                        
                mape=np.mean(np.mean(abs(np.array(y_test)-pred)/abs(np.array(y_test)),axis=0),axis=0)
            
            perf1.append(mape)
            
            frame=[max(w-2,1),w] # frame
            h=5 # horizon factor
            
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
            
            # LSTM model implementation
            model = Sequential()
            model.add(LSTM(units=128,activation='relu',input_shape=(None,1)))
            model.add(Dense(1))
            model.compile(loss='mae',optimizer='adam',metrics=['mae'])
            
            for i in range(n_epochs):
                for j in range(len(x_train)):
                    model.train_on_batch(x_train[j],y_train[j])
                              
                pred=[]
                for j in range(len(x_train)):    
                    pred.append(model.predict(x_test[j]))
                
                pred1=pred
                
                ave=np.mean(pred,axis=0)
                for k in range(len(pred)): # replace pred with average
                    pred[k]=ave

                mape=np.mean(np.mean(abs(np.array(y_test)-pred)/abs(np.array(y_test)),axis=0),axis=0)
            
            perf2.append(mape)
            print('MAPE(',w,')=\n',perf1,'\n',perf2)

            with open("perf_single_bitcoin1.txt","wb") as f:
                pickle.dump(perf1,f)
            with open("perf_single_bitcoin2.txt","wb") as f:
                pickle.dump(perf2,f)   
    
    A = np.array(perf1).reshape(-1,10).transpose() #np.random.rand(100,10)
    B = np.array(perf2).reshape(-1,10).transpose() #np.random.rand(100,10)
    
    def draw_plot(data,offset,edge_color,fill_color,label):
        pos = np.arange(data.shape[1])+offset  
        bp = ax.boxplot(data,positions=pos,widths=0.3,patch_artist=True,labels=label)
        for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color=edge_color)   
    
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)
        return bp    
           
    fig, ax = plt.subplots()
    bp1=draw_plot(A,-0.25,"red","white",['1/1','2/1','3/1','4/1','5/1','6/1','7/1','8/1','9/1','10/1','11/1','12/1'])
    bp2=draw_plot(B,+0.25,"blue","white",['1/1','2/2','3/3','4/3','5/3','6/3','7/3','8/3','9/3','10/3','11/3','12/3'])
    plt.title('Bitcoin price data (closing price)')
    plt.xticks(rotation=45,fontsize=8)
    plt.xlabel('w/v denoting R(w,v)')
    plt.ylabel('MAPE')
    plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['sparse sampling', 'dense sampling'], loc='best')    
    plt.grid()
    plt.show()
    plt.close()  
elif mode==2: # multi-steps mode
    perf1,perf2=[],[]
    
    for h in range(1,6):
        for s in range(10):
            w=h+1
            frame=[w,w] # frame
            
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
            
            # LSTM model implementation
            model = Sequential()
            model.add(LSTM(units=128,activation='relu',input_shape=(None,1)))
            model.add(Dense(1))
            model.compile(loss='mae',optimizer='adam',metrics=['mae'])
        
            for i in range(n_epochs):
                for j in range(len(x_train)):             
                    model.train_on_batch(x_train[j],y_train[j])
                    
                pred=[]
                for j in range(len(x_train)):    
                    pred.append(model.predict(x_test[j]))
            
                ave=np.mean(pred,axis=0)
                for k in range(len(pred)): # replace pred with average
                    pred[k]=ave
                        
                mape=np.mean(np.mean(abs(np.array(y_test)-pred)/abs(np.array(y_test)),axis=0),axis=0)
            perf1.append(mape)
            
            frame=[max(w-2,1),w] # frame
            
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
            
            # LSTM model implementation
            model = Sequential()
            model.add(LSTM(units=128,activation='relu',input_shape=(None,1)))
            model.add(Dense(1))
            model.compile(loss='mae',optimizer='adam',metrics=['mae'])
        
            for i in range(n_epochs):
                for j in range(len(x_train)):
                    model.train_on_batch(x_train[j],y_train[j])
                
                pred=[]
                for j in range(len(x_train)):    
                    pred.append(model.predict(x_test[j]))
                
                pred1=pred
                
                ave=np.mean(pred,axis=0)
                for k in range(len(pred)): # replace pred with average
                    pred[k]=ave
                        
                mape=np.mean(np.mean(abs(np.array(y_test)-pred)/abs(np.array(y_test)),axis=0),axis=0)
            perf2.append(mape)
            print('MAPE(',w,')=\n',perf1,'\n',perf2)

            with open("perf_multi_bitcoin1.txt","wb") as f:
                pickle.dump(perf1,f)
            with open("perf_multi_bitcoin2.txt","wb") as f:
                pickle.dump(perf2,f)   
    
    A = np.array(perf1).reshape(-1,10).transpose() #np.random.rand(100,10)
    B = np.array(perf2).reshape(-1,10).transpose() #np.random.rand(100,10)
    
    meanA=A.mean(axis=0)
    meanB=B.mean(axis=0)
    
    def draw_plot(data,offset,edge_color,fill_color,label):
        pos = np.arange(data.shape[1])+offset  
        bp = ax.boxplot(data,positions=pos,widths=0.3,patch_artist=True,labels=label)
        for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color=edge_color)   
    
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)
        
        # https://stackoverflow.com/questions/58066009/how-to-display-numeric-mean-and-std-values-next-to-a-box-plot-in-a-series-of-box
        for i, line in enumerate(bp['medians']):
            x, y = line.get_xydata()[1]
            if edge_color=='red':
                text = '{:.4f}'.format(meanA[i])
                ax.annotate(text,xy=(x-0.6,y),color='black',size=8)
            else:
                text = '{:.4f}'.format(meanB[i])
                ax.annotate(text,xy=(x,y),color='black',size=8)
        return bp          
           
    fig, ax = plt.subplots()
    bp1=draw_plot(A,-0.25,"red","white",['1','2','3','4','5'])
    bp2=draw_plot(B,+0.25,"blue","white",['1','2','3','4','5'])
    plt.title('Bitcoin price data (closing price)')
    plt.xticks(rotation=0,fontsize=8)
    plt.xlabel('h')
    plt.ylabel('MAPE')
    plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['sparse sampling', 'dense sampling'], loc='lower right')
    plt.grid()
    plt.show()
    plt.close()  
    # vertical grid lines/ https://stackoverflow.com/questions/61329069/add-vertical-lines-to-separate-condition-split-boxplots-using-seaborn
    