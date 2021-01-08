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

print(caffe.TEST)
net = caffe.Net(MODEL_FILE,caffe.TEST, weights=PRETRAINED)
caffe.set_device(0)
caffe.set_mode_gpu()

print("Caffe set")


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

os.chdir("PascalVOCresults")

for key, value in lmdb_cursor:
        print "Count:"
        print count
        count = count + 1
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        t_y.append(int(datum.label))
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        im = image#[0,i1_start:i1_stop, i2_start:i2_stop]
        out = net.forward_all(data=np.asarray([im]))
        p_y.append(out['prob'][0].argmax(axis=0))
        feat_tt.append(net.blobs['fc7_res'].data[0].tolist())
        print("Label is class " + str(int(datum.label)) + ", predicted class is " + str(out['prob'][0].argmax(axis=0)))
        
        outputFileName = "Correct.txt"
        #Increment associated counter
        if str(out['prob'][0].argmax(axis=0)) is '1':
            MFCount += 1
            outputFileName = "MedianFiltering.txt"
        elif str(out['prob'][0].argmax(axis=0)) is '2':
            GBCount += 1
            outputFileName = "GaussianBlurring.txt"
        elif str(out['prob'][0].argmax(axis=0)) is '3':
            WGNCount += 1
            outputFileName = "WhiteNoise.txt"
        elif str(out['prob'][0].argmax(axis=0)) is '4':
            RSCount += 1
            outputFileName = "Resampling.txt"
        elif str(out['prob'][0].argmax(axis=0)) is '5':
            JPGCount += 1
            outputFileName = "JPGCompression.txt"
            
        #Load output into out file
        outputFile = open(outputFileName, 'a')
        outputFile.write(key + "\n")
        outputFile.close()
            
print("Times predicted Median Filtering: ", MFCount)
print("Times predicted Gaussian Blurring: ", GBCount)
print("Times predicted Additive White Gaussian Noise: ", WGNCount)
print("Times predicted Resampling: ", RSCount)
print("Times predicted JPG Compression: ", JPGCount)

#####################################Softmax confusion matrix and testing accuracy
from sklearn.metrics import confusion_matrix

cmat = confusion_matrix(t_y, p_y)
nbr = cmat.sum(1)
nbr = np.array(nbr, dtype = 'f')
M = cmat/nbr
np.set_printoptions(suppress=True)
M = np.around(M*100, decimals=2) # set the confusion matrix to two decimals

binary = [t_y[i]==p_y[i] for i in range(len(p_y))]
acc = binary.count(True)/float(count)

print 'The testing accuracy is ' + str(acc)