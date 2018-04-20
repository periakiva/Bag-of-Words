# Bag-of-Words

## Requirements:
```
OpenCV 2.4 (I believe)
Numpy
glob
os
Matplotlib
Sklearn
scipy
mpl_toolkits
```

## Run with already loaded set:
```
$ git clone git@github.com:periakiva/Bag-of-Words.git
$ python predict.py
```

## Run with your own data:
```
$ git clone git@github.com:periakiva/Bag-of-Words.git
$ cd Bag-of-Words
$ mkdir train_my_own
$ cd train_my_own
```
At this point make a folder for each class and insert images + change path in code to your new training folder **/train_my_own/**
```
$ python feat.py
$ python features.py
$ python predict.py
```
