#!/usr/bin/env python
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
import lmfilters
import cv
import cv2
from numpy import *
import pymeanshift as pms

def meanshiftUsingIntensity(path):
	im = cv.LoadImageM(path,cv.CV_LOAD_IMAGE_GRAYSCALE)
	(segmentedImage, labelsImage, numberRegions) = pms.segmentMeanShift(im)
	print "number of region" , numberRegions
	return segmentedImage

def meanshiftUsingIntensityAndLocation(path):
    im = cv.LoadImageM(path,cv.CV_LOAD_IMAGE_GRAYSCALE)
    #creat a mat from the pixel intensity and its location
    mat = cv.LoadImageM(path)
    for i in xrange(0,im.height):
        for j in xrange(0,im.width):
            value = (im[i,j], i, j)
            mat[i,j] = value
	    print mat[i,j]

    (segmentedImage, labelsImage, numberRegions) = pms.segmentMeanShift(mat)

    clusters = {}
    for i in xrange(0,labelsImage.height):
        for j in xrange(0,labelsImage.width):
            v = labelsImage[i,j]
            if v in clusters:
                clusters[v].append(im[i,j])
            else:
                clusters[v] =  [im[i,j]]

    means = {}
    for c in clusters:
        means[c] = sum(clusters[c])/len(clusters[c])

    for i in xrange(0,im.height):
        for j in xrange(0,im.width):
            lbl = labelsImage[i,j]
            im[i,j] = means[lbl]

    print "number of region" , numberRegions
    return im
    #return segmentedImage

def meanshiftUsingRGB(path):
    im = cv.LoadImageM(path)
    (segmentedImage, labelsImage, numberRegions) = pms.segmentMeanShift(im)
    print "number of region" , numberRegions
    return segmentedImage

def meanshiftUsingYUV(path):
    im = cv.LoadImageM(path)
    cv.CvtColor(im,im,cv.CV_BGR2YCrCb)
    (segmentedImage, labelsImage, numberRegions) = pms.segmentMeanShift(im)
    print "number of region" , numberRegions
    return segmentedImage

# Return a filter type given its index
def getFilterTypeIndex(index):
	if index >=0 and index <= 17:
		return 0
	elif index >= 18 and index <= 35:
		return 1
	elif index >= 36 and index <= 45:
		return 2
	else:
		return 3

def meanshiftUsingILM(path):
	im = cv.LoadImageM(path)
	#array of filter kernels
	filterBank = lmfilters.loadLMFilters()
	resize_factor = 5
	thumbnail = cv.CreateMat(im.height / resize_factor, im.width / resize_factor, cv.CV_8UC3)
	cv.Resize(im, thumbnail)
	cv.ShowImage("small",thumbnail)
	X = np.zeros(shape=((thumbnail.height)*(thumbnail.width),4), dtype=float)
	for f in xrange(0,48):
		filter = filterBank[f]
		dst = cv.CreateImage(cv.GetSize(thumbnail), cv.IPL_DEPTH_32F, 3)
		thumbnail2 = cv.CreateMat(filter.height / resize_factor, filter.width / resize_factor, filter.type)
		cv.Resize(filter, thumbnail2)
		cv.Filter2D(thumbnail,dst,thumbnail2)
		cv.ShowImage("filter",dst)
		featureIndex = getFilterTypeIndex(f)
		for j in xrange(0,thumbnail.height):
			for i in xrange(0,thumbnail.width):
				maxRes = max(dst[j,i])
				if math.isnan(maxRes):
					maxRes = 0.0
				if maxRes > X[thumbnail.width*j+i,featureIndex]:
					X[thumbnail.width*j+i,featureIndex] = maxRes

	ms = MeanShift()
	ms.fit(X)
	labels = ms.labels_
	print labels
	n_clusters_ = np.unique(labels)
	print n_clusters_
	clusters = {}
	for i in xrange(0,thumbnail.height):
		for j in xrange(0,thumbnail.width):
			v = labels[thumbnail.width*i+j]
			if v in clusters:
				clusters[v].append(thumbnail[i,j])
			else:
 				clusters[v] =  [thumbnail[i,j]]

	means = {}
	for c in clusters:
		# new cluster
		red = 0.0
		green = 0.0
		blue = 0.0
		for p in clusters[c]:
			red = red + p[0]
			green = green + p[1]
			blue = blue + p[2]
		red = red / len(clusters[c])
		green = green / len(clusters[c])
		blue = blue / len(clusters[c])
		mytuple = red, green, blue
		means[c] = mytuple

	for i in xrange(0,thumbnail.height):
		for j in xrange(0,thumbnail.width):
 			lbl = labels[thumbnail.width*i+j]
			thumbnail[i,j] = means[lbl]

	#print "number of region" , numberRegions
	cv.Resize(thumbnail, im)
	return im




#name = "../test images/general/58060.jpg"
name = "../test images/single object/189080.jpg"

#im = meanshiftUsingIntensity(name)
im = meanshiftUsingILM(name)
#im = meanshiftUsingIntensityAndLocation(name)
#im = meanshiftUsingRGB(name)
#im = meanshiftUsingYUV(name)

cv.ShowImage("win1",im)
cv.WaitKey(0)
