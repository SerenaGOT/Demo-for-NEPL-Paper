Multi-View Intact Space Learning for Tinnitus Classification in Resting State EEG

This archive contains a python implementation of the method for Tinnitus EEG Segment Classification Based on Multi-view Intact Space Learning proposed in the paper.

Z.-R Sun, Y.-X Cai, S.-J Wang, C.-D Wang, Y.-Q Zheng

FILES INCLUDE:
1.Original Data Set
Our data set consist of 36 patients' 928 valid segments of EEG data, the whole dataset is 4.26G, it is too large to submit, so we are going into submit the file cotaining the extraction features which is much smaller. 
- small_data£ºThis folder includes the 2 segments namely sample1.txt and sample2.txt.
- feature: This folder includes the final multi-view feature file named "latent_feature.csv" and "latent_feature.npy".

2. Code Files
- signal_process.py
This file is to process the orginal data, and extract the features in different views, including time domain feature, time domain statistical feature and frequency domain feature. We provide a small dataset with 2 segments just to testify the program (in folder "small_data").£¨This file requires the environment of python3.5.£©
- multiview.py
This file is used to combine the features together. Its input  are the 3 different view and output is the final multi-view feature file into "latent_feature.csv" and "latent_feature.npy¡±£¨This file should run after running "signal_process.py" and requires the environment of python3.5£©
- classification.py
This file is to output the final classification result including the accuracy, precision, recall and f1-means. For this file, we would provide the whole combining features "feature/latent_feature.npy" and the label file "feature/labels_binary.csv¡±£¨file includes name of each segment, and corresponding index and class label. In class labels, 0 is for normal control and 1 is for tinnitus patient), it would output the real result of the total dataset.£¨This file requires the environment of python2.7.£©


PROPRIETIES OF DATASET
The properties of the dataset are summarized as followed.

Subject           Instances     Segments  Dimensionality of each Segment

Tinnitus patients    18 	  537		2000*129

Normal controls	     18		  391		2000*129


INSTALLATION AND REQUIREMENT
No installation is required. The code is tested on PyCharm (Version 2016.2.3) with python 2.7 and python 3.5 on macOS Sierra (64 bits, 1.4 GHz Intel Core i5, 4 GB 1600 MHz DDR3)


CONTACT
Changdong Wang
Sun Yat-Sen University, P. R. China 
changdongwang@live.cn; changdongwang@hotmail.com
