import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from scipy.cluster.vq import *
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets, metrics

#load the needed data
clf, features, classlist, slr, k, iterations = joblib.load("bof.pkl")

sift = cv2.xfeatures2d.SIFT_create()
testpath = glob.glob('/this/is/your/path/to/test/*.png')

#plot clusters
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
x,y,z = [],[],[]

for i in iterations:
	x.append(i[0])
	y.append(i[1])
	z.append(i[2])
	ax.scatter(x,y,z,zdir='z',s=100)

plt.show()

imagepaths=[]
classespath = "/this/is/your/path/to/test/"
deslist =[]
imageclasses=[]
paths=[]
classlabels=[]
classpaths=[]
classid=0
classlist=os.listdir(classespath)
print classlist

#label images
deslist1 = np.zeros(shape=(128,1))
for test in classlist:
	dir = os.path.join(classespath,test)
	classpath = os.listdir(dir)
	classpaths.append(dir)
	classlabel=test
	imageclasses+=[classlabel]*len(classpath)
	classlabels.append(classid)
	classid+=1

print imageclasses
print classpaths

#get descriptor vectos and append them together
for j in classpaths:
	p=glob.glob(j+"/*.png")	
	for img in p:
		print img
		paths.append(img)
		im = cv2.imread(img)
		gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(gray,None)
		deslist.append((des))

deslist = np.array(deslist)
imageclasses = np.asarray(imageclasses)
testfeatures = np.zeros((len(imageclasses),k),'float32')

for i in xrange(len(imageclasses)):
	words, distance = vq(deslist[i],iterations)
	for w in words:
		testfeatures[i][w] += 1

oc = np.sum( (testfeatures > 0)*1.0, axis=0)
idf = np.array(np.log((1.0*len(testpath)+1)/(1.0*oc+1)),'float32')
testfeatures = slr.transform(testfeatures)

h=0.02

print testfeatures.shape

#creat a prediction vector 
predictions = [classlist[i] for i in clf.predict(testfeatures)]

for test, prediction, i in zip(paths,predictions,xrange(len(imageclasses))):
	if prediction == imageclasses[i]:
		print "true"
	else:
		print "false"
	image = cv2.imread(test)
	cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
	pt = (0,3*image.shape[0] // 4)
	print prediction
	cv2.putText(image,prediction,pt,cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2,[0,255,0],2)
	cv2.imshow("image",image)
	cv2.waitKey(3000)

#confusion matrix
print("classification report for classifier %s:\n%s\n" % (clf,metrics.classification_report(imageclasses, predictions)))
c = confusion_matrix(imageclasses,predictions)
print "confusion matrix: "
print c
