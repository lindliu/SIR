#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:33:28 2022

@author: do0236li
"""

###https://github.com/covid19-eu-zh/covid19-eu-data

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


init = 0

path = os.path.join(os.getcwd(), './Covid_19/EU_dataset/covid-19-at.csv')
dataframe = pd.read_csv(path,sep=',')
dataframe = dataframe.dropna()
dataframe.columns

### data cleaning ###
data = dataframe.to_numpy()
for i in range(data.shape[0]):
    data[i,1] = data[i,1].replace('\xad','')
    data[i,-1] = int(data[i,-1][:10].replace('-',''))
    
print(np.unique(data[:,1]))
# print(np.where(data[:,1]!='Repatriierte'))
# data = data[data[:,1]!='Repatriierte',:]

covid_data = np.zeros([data.shape[0],5], dtype=np.int32)
states = np.unique(data[:,1])
states_dict = []
for state in states:
    states_dict.append((state, init))
    init += 1
states_dict = dict(states_dict)

for state,idx in states_dict.items():
    covid_data[data[:,1]==state, 0] = idx #index of area
    covid_data[data[:,1]==state, 1] = data[data[:,1]==state, 8] #time
    covid_data[data[:,1]==state, 2] = dataframe.loc[dataframe.nuts_2==state].cases #infected
    covid_data[data[:,1]==state, 3] = dataframe.loc[dataframe.nuts_2==state].recovered #recovered
    covid_data[data[:,1]==state, 4] = dataframe.loc[dataframe.nuts_2==state].deaths #death


covid_data_s = covid_data[np.argsort(covid_data[:,0]),:] #sorted by index of area
for value in states_dict.values():
    idx_ = np.where(covid_data_s[:,0]==value)[0]
    covid_data_s[idx_,:] = covid_data_s[idx_[np.argsort(covid_data_s[idx_, 1])],:]
    
# plt.plot(covid_data_s[covid_data_s[:,0]==6, 3])





## https://www.ecdc.europa.eu/en/covid-19/data
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

path = os.path.join(os.getcwd(), './Covid_19/data.csv')
dataframe = pd.read_csv(path,sep=',')
dataframe = dataframe.dropna()
dataframe.columns


data = dataframe.to_numpy()


covid_data = np.zeros([data.shape[0],4], dtype=np.int32)
country_dict = dict([(country, i) for i,country in enumerate(np.unique(data[:,8]))])
for count,value in country_dict.items():
    covid_data[np.where(data[:,8]==count),0] = value
    
covid_data[:,1] = [int(i[-4:]+i[-7:-5]+i[-10:-8]) for i in data[:,0]] #time
covid_data[:,2] = data[:,4] #cases
covid_data[:,3] = data[:,5] #deaths


covid_data = covid_data[covid_data[:,2]>=0]
covid_data = covid_data[covid_data[:,3]>=0]




syear = int(str(covid_data[:,1].min())[:4])
smonth = int(str(covid_data[:,1].min())[4:6])
sday = int(str(covid_data[:,1].min())[6:])
eyear = int(str(covid_data[:,1].max())[:4])
emonth = int(str(covid_data[:,1].max())[4:6])
eday = int(str(covid_data[:,1].max())[6:])

sdate = date(syear,smonth,sday)   # start date
edate = date(eyear,emonth,eday)   # end date
date_list = [int(i) for i in pd.date_range(sdate,edate-timedelta(days=1),freq='d').strftime('%Y%m%d').to_list()]
np.repeat(date_list,len(country_dict),axis=0)

covid_data_ = np.zeros([len(date_list)*len(country_dict),4], dtype=np.float32)
covid_data_[:,0] = np.repeat(list(country_dict.values()), len(date_list), axis=0)
covid_data_[:,1] = np.repeat(np.array(date_list).reshape([1,-1]), len(country_dict), axis=0).flatten()
for i in range(covid_data_.shape[0]):
    
    idx_state = np.where(covid_data[:,0]==covid_data_[i,0])[0]
    idx_date = np.where(covid_data[:,1]==covid_data_[i,1])[0]
    
    idx = list(set(idx_date).intersection(set(idx_state)))
    if len(idx) == 0:
        continue
    else:
        covid_data_[i,:] = covid_data[idx[0],:]

plt.figure(1)
idx=covid_data_[:,0]==10#country_dict['AUT']
plt.plot(covid_data_[idx, 2])


import copy
covid_data_m = copy.deepcopy(covid_data_)
covid_data_m[:,2:4] = 0
for state in country_dict.values():
    covid_data_m[covid_data_[:,0]==state,2] = np.convolve(np.ones(7)/7, covid_data_[covid_data_[:,0]==state,2],'same')
    covid_data_m[covid_data_[:,0]==state,3] = np.convolve(np.ones(7)/7, covid_data_[covid_data_[:,0]==state,3],'same')

plt.figure(2)
idx=covid_data_m[:,0]==7#country_dict['AUT']
plt.plot(covid_data_[idx, 2])
plt.plot(covid_data_m[idx, 2])



# covid_data_m[]


daily_cases = covid_data_m[:,2].reshape([-1,(covid_data_m[:,0]==0).sum()])
plt.figure(3)
plt.plot(daily_cases[7,:])

plt.figure(4)
plt.plot(daily_cases[:,1:]-daily_cases[:,:-1])



