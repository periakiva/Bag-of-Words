import cv2
import numpy as np
import glob
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from scipy.cluster.vq import *
from sklearn import svm, datasets
from sklearn import neighbors
from matplotlib.colors import ListedColormap
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model

#set parameters
sift = cv2.xfeatures2d.SIFT_create()
trainpath = "/home/periperi/school/rcv/project4/train/"
test = glob.glob('/home/periperi/school/rcv/test/*.png')
classeslist = os.listdir(trainpath)

desvector=[]
deslist=[]
classpaths = []
classlabels =[]
imageclasses=[]
classid=0

#label images
for train in classeslist:
	dir = os.path.join(trainpath,train)
	print dir
	classpath = os.listdir(dir)
	classpaths.append(dir)
	classlabel = train
	imageclasses+=[classlabel]*len(classpath)
	classlabels.append(classid)
	classid+=1

print "imageclasses"
print imageclasses
print "classlabels"
print classlabels
print "classpaths"
print classpaths
deslist1=np.zeros(shape =(128,1))
print "deslist1 shape"
print deslist1.shape
print "y shape"
print np.array([classlabels,1]).shape
imc = imageclasses
print imageclasses[2400]

imageclasses = np.asarray(imageclasses)
classis=0

#get descriptor vector of each image and append
for j in classpaths:
	p = glob.glob(j+"/*.png")
	print "now evaluating" + j
	counter=0
	for i in p:
		print i
		classis+=1
		counter+=1
		img = cv2.imread(i)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(gray,None)
		desvector.append((i,des))
		deslist.append((des))
		des = np.transpose(des)

		#in case you want to look at individual sample's histogram
		print classis
		if counter%200 == 0:
			cv2.drawKeypoints(gray,kp,img)

			cv2.imwrite(str(classis)+".jpg",img)
			#plt.hist(des,bins=xrange(8))
			#plt.title(imageclasses[classis])
			#plt.savefig("/home/periperi/school/rcv/project4/results/"+imageclasses[classis]+str(counter)+".png")
			#plt.close()
			
			print i		
	

		print i
		print des.shape
		deslist1 = np.concatenate((deslist1,des),axis=1)
		print len(desvector)
		print len(deslist)
		
deslist = np.array(deslist)
deslist1=np.transpose(deslist1)

#write a data file with the necessary data
joblib.dump((deslist1,imageclasses,deslist,classpaths,classeslist),"feat.pkl",compress=3)
