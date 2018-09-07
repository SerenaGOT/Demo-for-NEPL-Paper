# -*- coding: utf-8 -*-
############
# Tinnitus EEG Segment Classification Based on Multi-view Intact Space Learning
# Input: 
#    frequency_domain_data:  data in the view of frequency domain 
#    statistic_frequency_data: data in the view of statistic characters in frequency domain
#    statistic_time_data: data in the view of statistic characters in time domain
#    time_domain_data: data in the view of time domain
# Output:
#   X: feature matrix in intact latent space after multi-view learning
# Reference:
#   Z.-R Sun, Y.-X Cai, S.-J Wang, C.-D Wang, Y.-Q Zheng, Tinnitus EEG Segment Classification Based on Multi-view Intact Space Learning
# May,2017
# Contact:
#   Chang-Dong Wang with Sun Yat-sen University.
#   Email: changdongwang@live.cn; changdongwang@hotmail.com
############
# script used for generating the multi-view features
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

# read feature
def read_data(file_name):
    data = []
    with open(file_name) as f:
        for line in f:
            line = line.replace("[","")
            line = line.replace("]", "")
            word = line.split(',')
            word = list(map(eval,word))
            data.append(word)

    return np.array(data)
# multi view
def multi_view(z,d,c,c1,c2):
    n = z[0].shape[0]  # sample amount
    m = len(z)  # view number
    dv = []
    for i in range(len(z)):
        dv.append(z[i].shape[1])
    Wv = []
    Q_x = []

    # initialize x
    x = np.random.randn(d, n)  # get the initial random data
    # initialize W
    for i in range(m):
        W = np.random.randn(dv[i], d)
        Wv.append(W)
    # initialize r and Q
    for i in range(m):
        r = z[i] - (Wv[i] @ x).T
        Q_x.append(1.0/(c**2+np.linalg.norm(r)))

    pre_total = pre_loss_x = 1
    total = loss_x = 0
    k = 1
    while(pre_total-total>1e-5 or k<15  and k <25):
        # iteration for x
        pre_total = total
        pre_loss_x = loss_x
        print("iteration",k)
        k+=1
        front = back = 0
        for v in range(m):
            front += (Wv[v].T * Q_x[v]) @ Wv[v]
            # print(front.shape)
            back += (Wv[v].T * Q_x[v]) @ z[v].T
        front += m*c2*np.eye(d)
        front = np.linalg.inv(front)
        x = front @ back
        print("x",x.shape)
        loss_x = 0
        total = 0
        for v in range(m):
            loss_x += math.log(1+np.linalg.norm(z[v]-(Wv[v]@x).T)/c,2)
        loss_x = loss_x/m + c2*np.linalg.norm(x)
        total += loss_x
        print("loss_x",loss_x)
        # update Q and r
        Q_x = []
        for i in range(m):
            r = z[i] - (Wv[i] @ x).T
            Q_x.append(1.0 / (c ** 2 + np.linalg.norm(r)))

        # iteration for W
        for v in range(m):
            Q_w = []
            front = back = 0
            for i in range(n):
                r = z[v][i,:] - (Wv[v] @ x[:,i]).T
                Q_w.append(1.0 / (c ** 2 + np.linalg.norm(r)))
                front += (z[v][i].reshape(-1,1) * Q_w[i]) @ x[:,i].reshape(1,-1)
                back += (x[:,i].reshape(-1,1) * Q_w[i]) @ (x[:,i].T).reshape(1,-1)
            back += n*c1*np.eye(d)
            back = np.linalg.inv(back)
            Wv[v] = front @ back
            loss_w = 0

            for i in range(n):
                loss_w += math.log(1 + np.linalg.norm(z[v] - (Wv[v] @ x[:,i]).T) / c, 2)
            loss_w = loss_w / n + c1 * np.linalg.norm(Wv[v])
            total += loss_w
            print("view",v,"wv_loss",loss_w)
        print("total loss",total)
    print(x.shape)
    return x.T


# normalize
def normalize(channel_data):
    channel_data = preprocessing.scale(channel_data)
    return channel_data


if __name__ == "__main__":  
    frequency_domain_data = read_data("frequency_domain_data.csv")
    statistic_frequency_data =read_data("statistic_frequency_data.csv")
    statistic_time_data = read_data("statistic_time_data.csv")
    time_domain_data = read_data("time_domain_data.csv")
    print("done reading file..")

    #normalize
    frequency_domain_data = normalize(frequency_domain_data)
    statistic_frequency_data = normalize(statistic_frequency_data)
    statistic_time_data = normalize(statistic_time_data)
    time_domain_data = normalize(time_domain_data)
    print("finish normalizing file..")

    z = [statistic_time_data, time_domain_data,frequency_domain_data]
    c = 1;
    c1 = 0.06 ;
    c2 = 0.001
    for i in range(1):
        X = multi_view(z,i,c,c1,c2)
        print(X)
        np.savetxt("latent_feature.csv",X, delimiter=',', fmt="%10.5f")
        np.save("latent_feature.npy",X)
    print("finishing constructing multi-view data.")