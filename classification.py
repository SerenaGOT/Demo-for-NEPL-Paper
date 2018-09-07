# -*- coding: utf-8 -*-
############
# Tinnitus EEG Segment Classification Based on Multi-view Intact Space Learning
# Input: 
#   full_x: the feature matrix for all segments
#   full_y: class label for each segment
# Output:
#   final_result: contain the final classification result of accuracy, recall, precision, F1
# Reference:
#   Z.-R Sun, Y.-X Cai, S.-J Wang, C.-D Wang, Y.-Q Zheng, Tinnitus EEG Segment Classification Based on Multi-view Intact Space Learning
# May,2017
# Contact:
#   Chang-Dong Wang with Sun Yat-sen University.
#   Email: changdongwang@live.cn; changdongwang@hotmail.com
############
# script use for classification with different method#
import numpy as np
import scipy.io as scio
import h5py
import sklearn as sk
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
import re
import math
from scipy import io



#read samples and labels
def load_mat_data(vec_file_name,label_file_name):
    data_mat = scio.loadmat(vec_file_name) #feature
    label_mat = scio.loadmat(label_file_name)
    full_y = label_mat["label"]
    full_x = data_mat["alpha1_power"]

    full_x = full_x.T  # transver
    full_y = full_y.reshape(full_y.shape[0],) # transform to one dimension
    return full_x,full_y

# read csv file, get features and labels
def read_csv_data(vec_file_name,label_file_name):
    full_x = np.load(vec_file_name)
    lines = np.loadtxt(label_file_name, delimiter=',', dtype='str')
    full_y = lines[1:,2].astype('int')
    return full_x,full_y

#get train set and test set
def split_data(full_x,full_y):
    x_train, x_test, y_train, y_test = train_test_split(full_x, full_y, test_size = 0.1, random_state = 0)
    return x_train, x_test, y_train, y_test


#svm for cross validation
def svm_method_partial(x_train, x_test, y_train, y_test,k):
    pca = PCA(n_components=k, whiten=True)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    svc = svm.SVC(kernel='rbf', C=41)  #  'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    svc.fit(x_train, y_train)
    pre = svc.predict(x_test)
    score = get_prediction(pre, y_test)
    return score
    

# calculate the classification result
def get_prediction(prediction,y_true):
    TP = FN = FP = TN = 0
    for i in range(len(prediction)):
        if prediction[i] == 1 and y_true[i] == 1:
            TP += 1
        elif prediction[i] == 0 and y_true[i] == 1:
            FN += 1
        elif prediction[i] == 1 and y_true[i] == 0:
            FP += 1
        elif prediction[i] == 0 and y_true[i] == 0:
            TN += 1
    accuracy = float(TP + TN) / float(TP + FN + FP + TN)
    recall = 1.0 * TP / (TP + FN)
    precision = 1.0 * TP / (TP + FP)
    F1 = (2.0 * precision * recall) / (precision + recall)
    FPR = FP *1.0/(FP+TN)
    sensitivity = TN*1.0/(FP+TN)
    FNR = FN*1.0/(FN+TP)
    specificity = TP*1.0/(FN+TP)
    res = [accuracy,recall,precision,F1]
    return res





# read the classification labels and the corresponding indices
def get_classification_labels(label_file_name):

    lines = np.loadtxt(label_file_name, delimiter=',', dtype='str')
    name_label = lines[1:,0].tolist()
    for i in range(len(name_label)):
        reobj = re.compile('standard\w+')
        name_label[i] = reobj.sub('',name_label[i])

    unique_name_index = sorted(set(name_label),key=name_label.index) 
    name_index = []
    full_y = lines[1:,2].astype('int')
    for name in unique_name_index:
        cur_index = [i for i,value in enumerate(name_label) if name == value]
        name_index.append([min(cur_index),max(cur_index)
                              ,full_y[cur_index[0]]])
    return name_index

def get_random_segment_set(full_x,full_y,num): # num 为测试集要取的分割份数
    split_index = 0
    for i in range(len(full_y)):   # get the index position of two classes，split_index is the first index corresponding to value 1
        if(full_y[i] == 1):
            split_index = i
            break
    sample_index = random.sample(range(int(split_index)), int(math.ceil(split_index/num)))
    sample_index.extend(random.sample(range(int(split_index)+1,len(full_y)),int(math.ceil((len(full_y)-split_index)/num))))
    # 划分数据集为训练集和测试集
    sample_index.sort()
    all_index = range(len(full_y))
    train_index =  list(set(all_index) - set(sample_index))
    train_x = [];train_y = [];test_x = []; test_y = [];
    for i in train_index:
        train_x.append(full_x[i])
        train_y.append(full_y[i])
    for i in sample_index:
        test_x.append(full_x[i])
        test_y.append(full_y[i])
    return np.array(train_x),np.array(train_y),np.array(test_x),np.array(test_y)


# seperate dataset into train set and test set
def split_people_set(sample,name_index,full_x,full_y):
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        x = full_x.tolist()
        y = full_y.tolist()
        for i in range(len(name_index)):
            if i in sample: # seperate for test set
                for j in range(name_index[i][0],name_index[i][1]+1):
                    test_x.append(x[j])
                    test_y.append(y[j])
            else:   # seperate for train set
                for j in range(name_index[i][0], name_index[i][1] + 1):
                    train_x.append(x[j])
                    train_y.append(y[j])
        return np.array(train_x),np.array(train_y),np.array(test_x),np.array(test_y)


# read seperated data without the feature before merging

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

# classification based on the separated view of feature
def get_seperate_feature_result(full_y,name_index,all_test_example):
    frequency_domain_data = read_data("preprocessing_pca75/frequency_domain_data.csv")
    statistic_frequency_data = read_data("preprocessing_pca75/statistic_frequency_data.csv")
    statistic_time_data = read_data("preprocessing_pca75/statistic_time_data.csv")
    time_domain_data = read_data("preprocessing_pca75/time_domain_data.csv")
    print("done reading file..")
    # x_train, x_test, y_train, y_test = split_data(frequency_domain_data, full_y)
    for k in range(10, 141,10):
        res_score = []
        for i in range(100):
            train_x, train_y, test_x, test_y = get_random_segment_set(time_domain_data,full_y,num=10)
            score = svm_method_partial(train_x, test_x, train_y, test_y, k)  # randomly seperate the data set
            res_score.append(score)
        res_score = np.array(res_score)
        print(res_score.sum(axis=0)/res_score.shape[0])


if __name__ == '__main__':

    full_x, full_y = read_csv_data("feature/latent_feature.npy", "feature/labels_binary.csv")
    name_index = get_classification_labels("labels_binary.csv")
    f = open("svm_pca_res.txt", "w")
    for k in range(130,131,1):
        res_score = []
        break
        for i in range(100): # get results of 100 times 
            train_x, train_y, test_x, test_y = get_random_segment_set(full_x,full_y,num=10)
            score = svm_method_partial(train_x, test_x, train_y, test_y, k)  # pick random segments 
            res_score.append(score)
        res_score = np.array(res_score)
        final_result = res_score.sum(axis=0) / res_score.shape[0]
        print "Accuracy Recall Precision F1"+"\n"
        print str(final_result[0])+" "+str(final_result[1])+" "+str(final_result[2])+" "+str(final_result[3])
 
   

   
