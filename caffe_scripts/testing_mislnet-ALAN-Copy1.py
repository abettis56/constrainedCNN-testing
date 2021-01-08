"""
             _       _                          
  _ __ ___  (_) ___ | | multimedia &                
 | '_ ` _ \ | |/ __|| | information
 | | | | | || |\__ \| | security
 |_| |_| |_||_||___/|_| lab
 __________________________________________________
|__________________________________________________|

 misl.ece.drexel.edu

 DEPT. OF ELECTRICAL & COMPUTER ENGINEERING
 DREXEL UNIVERSITY
"""

import os,math
import sys
import caffe
import cv2
import numpy as np
from numpy import *
import lmdb
from sklearn import svm
import matplotlib.pyplot as plt
plt.switch_backend('agg')
caffe_root = '/home/,,/,,/opt/caffe'



MODEL_FILE = 'deploy_mislnet.prototxt' #Make sure about the path where you saved your prototxt file
PRETRAINED = 'mislnet_six_classes.caffemodel' #Make sure about the path where you saved your caffe model
"""
print(caffe.TEST)
net = caffe.Net(MODEL_FILE,caffe.TEST, weights=PRETRAINED)
caffe.set_device(0)
caffe.set_mode_gpu()

print("Caffe set")
"""

lmdb_env = lmdb.open('test_lmdb_labeled') #Make sure about the path where you saved the test_lmdb data

lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
count = 0
correct = 0

p_y = []
t_y = []
feat_tt = []

#In case you need to test a trained CNN with smaller patches, e.g., 64x64
########################################
n = 64
i1_start = (256-n)/2
i1_stop = i1_start + n
i2_start = (256-n)/2
i2_stop = i2_start + n
########################################
#MF (median filtering)
MFCount = 0
#GB (gaussian blurring)
GBCount = 0
#WGN (additive white gaussian noise)
WGNCount = 0
#RS (resampling with bilinear interpolation)
RSCount = 0
#JPG (JPG compression)
JPGCount = 0

for key, value in lmdb_cursor:
        print "Count:"
        print count
        count = count + 1
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        t_y.append(int(datum.label))
        image = caffe.io.datum_to_array(datum)
        print(key)
        image = image.astype(np.uint8)
        im = image#[0,i1_start:i1_stop, i2_start:i2_stop]