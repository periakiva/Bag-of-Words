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

sift = cv2.xfeatures2d.SIFT_create()
#load necessary data
deslist1,imageclasses,deslist,classpaths,classeslist = joblib.load("feat.pkl")

#defining number of clusters
k=25
iterations, variance = kmeans(deslist1, k,1)

#some debugging and validation steps
#print deslist.shape
#print len(iterations)
#print iterations.shape
#print deslist1.shape
#print len(imageclasses)

#label images for comparison later on
#those are the histograms for each image to create bag of visual words
features = np.zeros((len(imageclasses),k),'float32')
for i in xrange(len(imageclasses)):
	words, distance = vq(deslist[i],iterations)
	histogramofw, binedges = np.histogram(words,bins=range(iterations.shape[0]+1),normed=True)
#	print words	
	#plt.hist(words,bins=range(iterations.shape[0]))
	#plt.show()
	for w in words:
		features[i][w] += 1
#		print w

#Those printing statement are for debugging purpuses, in case any of you need to.
#print "features shape"
#print features.shape
#print "deslist1 shape"
#print deslist1.shape
#print "deslist shape"
#print deslist.shape
#print len(deslist)
#print "image classes size"
#print len(imageclasses)

#transform vector
oc = np.sum((features>0)*1,axis=0)
idf = np.array(np.log((1.0*len(classpaths)+1)/(1.0*oc+1)),'float32')
slr = StandardScaler().fit(features)
features = slr.transform(features)

C=1.0
h = 0.02

print "features shape"
print features.shape
print "features"
print features

#The different types of predictions possible for different types of data. 
#Here I chose rbf because of the little features letters and digits have.

#svc = svm.SVC(kernel='linear',C=1.0).fit(featuresplot,np.array(classlabels))
#rbf_svc = svm.SVC(kernel='rbf',gamma=0.7,C=C).fit(featuresplot,np.array(classlabels))
#poly_svc = svm.SVC(kernel='poly',degree=3,C=C).fit(featuresplot,np.array(classlabels))
#lin_svc = svm.LinearSVC(C=C).fit(featuresplot,np.array(classlabels))
#clf = svm.SVC(kernel='poly',degree=3,C=C)
#clf = svm.LinearSVC(C=C)
#clf = neighbors.KNeighborsClassifier(k,weights='uniform',p=2,metric='minkowski')
#clf = neighbors.KNeighborsClassifier()
clf = svm.SVC(kernel='rbf',gamma=0.15,C=C)
clf.fit(features, np.array(imageclasses))

#write all data necessary for prediction
joblib.dump((clf,features,classeslist,slr,k,iterations),"bof.pkl",compress=3)



#cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
#cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF'])
#for weights in ['uniform','distance']:
#clf.fit(features,np.array(imageclasses))

#this part is to visualize the calssification. Fell free to use
"""
markers =('s','x','o','^','v')
colors = ('red','blue','lightgreen','gray','cyan')
cmap = ListedColormap(colors[:len(np.unique(np.array(classlabels)))])
xmin, xmax = features[:,0].min()-1,features[:,0].max()+1
ymin,ymax = features[:,1].min()-1,features[:,1].max()+1

xx,yy = np.meshgrid(np.arange(xmin,xmax,h),np.arange(ymin,ymax,h))
print xx
print yy
Z = clf.predict(np.array([xx.ravel(),yy.ravel()]))
Z = Z.reshape(xx.shape)
plt.figure()
plt.contourf(xx,yy,Z,alpha=0.4,cmap=camp)
plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
plt.scatter(featuresplot[:,0],featuresplot[:,1],cmap=cmap_bold)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
	
plt.show()
"""
"""


for i, clf in enumerate((svc,lin_svc,rbf_svc,poly_svc)):
	plt.subplot(2,2,i+1)
	plt.subplots_adjust(wspace=0.4,hspace=0.4)
	plt.contourf(xx,yy,Z,cmap=plt.cm.coolwarm,alpha=0.75)
	plt.scatter(featuresplot[:,0],featuresplot[:,1],c=np.array(classlabels),cmap=plt.cm.coolwarm)
plt.show()

fig = plt.figure()
ax=fig.add_subplot(111,projection='3d')
x,y,z = [],[],[]

for i in iterations:
	x.append(i[0])
	y.append(i[1])
	z.append(i[2])
	ax.scatter(x,y,z,zdir='z',s=100)

plt.show()
"""
