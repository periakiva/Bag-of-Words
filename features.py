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

#this part is now a seperate function for efficency purposes
#i kept it here for easy access
"""
trainpath = "/home/periperi/school/rcv/project4/train/"
test = glob.glob('/home/periperi/school/rcv/test/*.png')
classeslist = os.listdir(trainpath)

desvector=[]
deslist=[]
classpaths = []
classlabels =[]
imageclasses=[]
classid=0
for train in classeslist:
	dir = os.path.join(trainpath,train)
	print dir
	classpath = os.listdir(dir)
	classpaths.append(dir)
	classlabel = train
	imageclasses+=[classid]*len(classpath)
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
for j in classpaths:
	p = glob.glob(j+"/*.png")
	print "now evaluating" + j
	for i in p:
		print i
		img = cv2.imread(i)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(gray,None)
	#	plt.hist(des,bins=range(8))
	#	plt.show()
		#plt.close(1)
		desvector.append((i,des))
		deslist.append((des))
		des = np.transpose(des)
		print i
		#attempt to concatenate to empty numpy
		print des.shape
		deslist1 = np.concatenate((deslist1,des),axis=1)
		#attempt to append to a list -> causing a list of numpy arrays, not sure if good or not
		print len(desvector)
		#print len(des)
		print len(deslist)
		#print deslist1.shape
		#print len(deslist1)
		
#deslist = np.asarray(deslist)
#descriptors = whiten(descriptors)
#print deslist.shape
#print deslist1
#print deslist1[:,0]
deslist = np.array(deslist)
imageclasses = np.asarray(imageclasses)
		

print len(deslist[0][0])
print deslist[0][0]
descriptors = np.array(deslist)
descriptors = whiten(descriptors)

print descriptors.size
print len(descriptors[0][0])
print descriptors.shape
#descriptors = np.transpose(descriptors)
print len(descriptors[0][0])
print len(descriptors[296][:])
print descriptors.shape
descriptors = whiten(deslist)
print "descriptors"
print descriptors
print len(descriptors)
print descriptors.shape
print len(descriptors[0][128])
print descriptors[0][128]

print desvector
print deslist
descriptors = deslist
for desc in deslist[1:]: 
	descriptors = np.vstack((descriptors,desc))


descriptors = desvector[0][1]
for img, desc in desvector[1:]:
	descriptors = np.vstack((descriptors,desc))

deslist1=np.transpose(deslist1)
joblib.dump((deslist1,imageclasses,deslist,classpaths,classlist),"feat.pkl",compress=3)
"""


sift = cv2.xfeatures2d.SIFT_create()
#load necessary data
deslist1,imageclasses,deslist,classpaths,classeslist = joblib.load("feat.pkl")

#defining number of clusters
k=25
iterations, variance = kmeans(deslist1, k,1)

#some debugging and validation steps
print deslist.shape
print len(iterations)
print iterations.shape
print deslist1.shape
print len(imageclasses)

#label images for comparison later on
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


print "features shape"
print features.shape
print "deslist1 shape"
print deslist1.shape
print "deslist shape"
print deslist.shape
print len(deslist)
print "image classes size"
print len(imageclasses)

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
