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

import os
from os import listdir
from random import shuffle
import cv2
from numpy import *
import numpy as np

import lmdb
import caffe
import scipy.io as sio

from scipy import misc
import cPickle



if __name__ == '__main__':
	#os.chdir('/home/belhassen/caffe_scripts')
	ll = cPickle.load(open('datasetOutput.dmp', 'r')) #load the list of images that have been used for the testing. These images have never been used for training
	os.chdir('/data/Forensics/PascalVOC/VOCdevkit/VOC2012/JPEGImages') #Change work directory to where you saved your Dresden images
	
	n = 1280 #cropping height 256x5
	m= 1280 # cropping width 256x5
	
	X = np.zeros((17125,1,256,256), dtype=np.uint8) #Initialize image data with zeros 
	y = np.zeros(17125, dtype=np.int64) #Initialize image labels with zeros
	count = 0
	for p in ll:
		img = cv2.imread(p)
		originalImage = img
		shape1 = 0
		if originalImage.shape[0] <= 256:
			shape1 = originalImage.shape[0]
		else:
			shape1 = 256
		
		shape2 = 0
		if originalImage.shape[1] <= 256:
			shape2 = originalImage.shape[1]
		else:
			shape2 = 256
		
		
		i1_start = np.uint8((originalImage.shape[0]-shape1)/2) #Find the coordinates of the central 1280x1280 sub-region
		i1_stop = i1_start + shape1
		i2_start = np.uint8((originalImage.shape[1]-shape2)/2)
		i2_stop = i2_start + shape2 
		truncatedImage = originalImage[i1_start:i1_stop, i2_start:i2_stop,:]
		
		imagePatch = np.zeros((256,256, 1), dtype=np.uint8)
		
		i1_start = np.uint8((imagePatch.shape[0]-shape1)/2) #Find the coordinates of the central 1280x1280 sub-region
		i1_stop = i1_start + shape1
		i2_start = np.uint8((imagePatch.shape[1]-shape2)/2)
		i2_stop = i2_start + shape2 
		imagePatch[i1_start:i1_stop, i2_start:i2_stop,0] = truncatedImage[:,:,1]
		X[count-1] = imagePatch.reshape((1,1,256,256))
		y[count-1] = 0
		count += 1
		print("Loaded image ",count)
	print("HI")
	from sklearn.utils import shuffle
	X, y = shuffle(X, y, random_state=0) #Shuffling data and labels with the same order. This is optional for testing and validation datasets
	
	print("BYE")
	os.chdir('/home/jupyter-alb719/ConstrainedCNN/constrained-conv-TIFS2018-master/caffe_scripts') #Change the work directory under where you are going to save the test_lmdb data
	
	N = X.shape[0]
	
	map_size = X.nbytes * 10
	
	env = lmdb.open('test_lmdb', map_size=map_size) #Create test_lmdb folder
	
	with env.begin(write=True) as txn:
		for i in range(N):
			datum = caffe.proto.caffe_pb2.Datum()
			datum.channels = X.shape[1]
			datum.height = X.shape[2]
			datum.width = X.shape[3]
			datum.data = X[i].tobytes()
			datum.label = int(y[i])
			str_id = '{:08}'.format(i)
			txn.put(str_id.encode('ascii'), datum.SerializeToString())
			print i+1