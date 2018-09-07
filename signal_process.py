# -*- coding: utf-8 -*-
############
# Tinnitus EEG Segment Classification Based on Multi-view Intact Space Learning
# Input: 
#    path: the root path of the file dictionary that needs to be tranversed
# Output:
#    frequency_domain_data:  data in the view of frequency domain 
#    statistic_frequency_data: data in the view of statistic characters in frequency domain
#    statistic_time_data: data in the view of statistic characters in time domain
#    time_domain_data: data in the view of time domain
# Reference:
#   Z.-R Sun, Y.-X Cai, S.-J Wang, C.-D Wang, Y.-Q Zheng, Tinnitus EEG Segment Classification Based on Multi-view Intact Space Learning
# May,2017
# Contact:
#   Chang-Dong Wang with Sun Yat-sen University.
#   Email: changdongwang@live.cn; changdongwang@hotmail.com
############
# script used for extract the origninal feature
import pandas as pd
import numpy as np
import os
import sklearn as sk
import pywt
import math
from sklearn.decomposition import PCA
import copy
from sklearn import preprocessing
from scipy.stats import mode



# view every file and get the feature of each segment
def file_transform(path):
    flag = 0
    result = np.array([])
    file1 = open("statistic_time_data.csv","w")
    file2 = open("time_domain_data.csv", "w")
    file3 = open("statistic_frequency_data.csv", "w")
    file4 = open("frequency_domain_data.csv", "w")
    for (path, dirs, files) in os.walk(path):  # tranverse the file directory.

        for filename in files:
            if(filename!=".DS_Store" and filename!=".DS_Store.csv"):

                print(filename, flag)
                flag+=1
                channel_data = read_data(path,filename)
                print("file1 read")
                channel_data = normalize(channel_data)  #normalize
                # get statistic time-domian data
                statistic_time_data = statistics(channel_data)
                statistic_time_data = normalize(statistic_time_data)
                statistic_time_data = merge_feature(statistic_time_data)
                file1.write(str(statistic_time_data.tolist()) + '\n')

                # get time-domian data
                time_domain_data = pca_method(channel_data)
                time_domain_data = normalize(time_domain_data)
                time_domain_data = merge_feature(time_domain_data)
                file2.write(str(time_domain_data.tolist()) + '\n')

                # get statistic frequency-domian data
                frequency_data = fft_transform(channel_data)
                statistic_frequency_data = statistics(frequency_data)
                statistic_frequency_data = statistics(statistic_frequency_data)
                statistic_frequency_data = merge_feature(statistic_frequency_data)
                file3.write(str(statistic_frequency_data.tolist()) + '\n')

                # get frequency-domian data
                frequency_domain_data = pca_method(frequency_data)
                frequency_domain_data = normalize(frequency_domain_data)
                frequency_domain_data = merge_feature(frequency_domain_data)
                file4.write(str(frequency_domain_data.tolist()) + '\n')
                
    file1.close()
    file2.close()
    file3.close()
    file4.close()


# read data from files
def read_data(path,filename):
    channel_data = []
    with open(path+'/'+filename,"r") as f:
        for line in f:
            word = line.split('\t')
            word = list(map(eval,word))
            channel_data.append(word)
        channel_data = np.array(channel_data)
    return channel_data.T


# get mode value
def get_mode(arr):
    mode = [];
    arr2 = arr.tolist()
    for arr in arr2:
        arr_appear = dict((a, arr2.count(a)) for a in arr);  
        if max(arr_appear.values()) == 1: 
            return;  
        else:
            for k, v in arr_appear.items():  
                if v == max(arr_appear.values()):
                    mode.append(k);
                    break
    return np.array(mode);

# statistic features
def statistics(channel_data):
    result = []
    maxx = channel_data.max(axis = 1).reshape(-1,1) # maximum value of each row
    minn = channel_data.min(axis = 1).reshape(-1,1) # minimum value of each row
    meann = channel_data.mean(axis = 1).reshape(-1,1) # mean value of each row
    midd = np.median(channel_data,axis = 1).reshape(-1,1) # median value of
    # varr = channel_data.var(axis = 0) # variance value of each row
    stdd = channel_data.std(axis = 1).reshape(-1,1) # standard difference of each row
    sort_data = copy.deepcopy(channel_data)
    sort_data.sort(axis=1)
    Xq1 = sort_data[:,math.floor((sort_data.shape[1])/4)].reshape(-1,1)  # first quartile
    Xq3 = sort_data[:,math.floor((sort_data.shape[1])*3/4)].reshape(-1,1) # third quartile
    range_value = (maxx - minn).reshape(-1,1)
    # mode_value = mode(channel_data,axis=1)[0].reshape(-1,1) # mode value of each row
    mode_value = get_mode(channel_data).reshape(-1,1)
    result = np.hstack((maxx,minn,meann,midd,stdd,Xq1,Xq3,range_value,mode_value))
    return result

# fft
def fft_transform(channel_data):
    fft_result = np.fft.fft2(channel_data)
    n = len(fft_result)
    fft_result = abs(fft_result)*1.0/n #normalize
    return fft_result


# normalize
def normalize(channel_data):
    # (X-mean)/std 
    channel_data = preprocessing.scale(channel_data)
    return channel_data

#pca
def pca_method(channel_data):
    pca = PCA(n_components=75, whiten=True)
    x_train = pca.fit_transform(channel_data)
    return x_train

# merge feature into one dimension  -- maybe promote to use multiview,
def merge_feature(channel_data):
    channel_data = channel_data.reshape(1,-1)
    return channel_data

# multi view
def multi_view(z):
    x_row = z[0].shape[0]
    dv = z[0].shape[1]
    Wv = []
    Q = []
    m = len(z)
    #initialize x
    x = np.random.randn(x_row, dv)  # get random initial dta
    c = 1
    c2 = 1
    # initialize W
    for i in range(m):
        W = np.random.randn(x_row, dv)
        Wv.append(W)
    # initialize r and Q
    for i in range(m):
        r = z[i] - Wv[i]*x
        Q.append(1.0/(c**2+np.linalg.norm(r)))
    # iteration
    for k in range(100):
        front = back = 0
        for v in range(m):
            front += Wv[v].T * Q[v] * Wv[v]
            print(front.shape)
            back += Wv[v].T * Q[v] * z[v]
        front += m*c2
        front = np.linalg.inv(front)
        x = front * back
        # update Q and rs
        Q = []
        for i in range(m):
            r = z[i] - Wv[i] * x
            Q.append(1.0 / (c ** 2 + np.linalg.norm(r)))
    print(x.shape)


if __name__ == "__main__":
    file_transform("small_data/")  # maybe change to the absolute direct
