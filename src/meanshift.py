#!/usr/bin/env python
from pca import *
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

# Repaint the image after segmentation such that all pixels in the same
# cluster are painted with the same color
def repaintImage(image,labels):
	# map from cluster to corresponding pixels
	clusters = {}
	for i in xrange(0,image.height):
		for j in xrange(0,image.width):
			v = labels[image.width*i+j]
			if v in clusters:
				clusters[v].append(image[i,j])
			else:
 				clusters[v] =  [image[i,j]]

	# map from the cluster type to the mean color for this cluter
	means = {}
	for c in clusters:
		# RGB colors for a given cluster
		color = [0.0, 0.0, 0.0]
		for cluster in clusters[c]:
			color[0] += cluster[0]
			color[1] += cluster[1]
			color[2] += cluster[2]
		color[0] = color[0] / len(clusters[c])
		color[1] = color[1] / len(clusters[c])
		color[2] = color[2] / len(clusters[c])
		mytuple = tuple(color)
		means[c] = mytuple

	# now repaint the image with the new color
	for i in xrange(0,image.height):
		for j in xrange(0,image.width):
 			lbl = labels[image.width*i+j]
			image[i,j] = means[lbl]

def meanshiftUsingILM(path):
	# Load original image given the image path
	im = cv.LoadImageM(path)
	# Load bank of filters
	filterBank = lmfilters.loadLMFilters()
	# Resize image to decrease dimensions during clustering
	resize_factor = 1
	thumbnail = cv.CreateMat(im.height / resize_factor, im.width / resize_factor, cv.CV_8UC3)
	cv.Resize(im, thumbnail)
	# now work with resized thumbnail image
	response = np.zeros(shape=((thumbnail.height)*(thumbnail.width),4), dtype=float)
	for f in xrange(0,48):
		filter = filterBank[f]
		# Resize the filter with the same factor for the resized image
		dst = cv.CreateImage(cv.GetSize(thumbnail), cv.IPL_DEPTH_32F, 3)
		resizedFilter = cv.CreateMat(filter.height / resize_factor, filter.width / resize_factor, filter.type)
		cv.Resize(filter, resizedFilter)
		# Apply the current filter
		cv.Filter2D(thumbnail,dst,resizedFilter)
		featureIndex = getFilterTypeIndex(f)
		for j in xrange(0,thumbnail.height):
			for i in xrange(0,thumbnail.width):
				# Select the max. along the three channels
				maxRes = max(dst[j,i])
				if math.isnan(maxRes):
					maxRes = 0.0
				if maxRes > response[thumbnail.width*j+i,featureIndex]:
					# Store the max. response for the given feature index
					response[thumbnail.width*j+i,featureIndex] = maxRes

	# Create new mean shift instance
	ms = MeanShift(bandwidth=10,bin_seeding=True)
	# Apply the mean shift clustering algorithm
	ms.fit(response)
	labels = ms.labels_
	n_clusters_ = np.unique(labels)
	print "Number of clusters: ", len(n_clusters_)
	repaintImage(thumbnail,labels)
	cv.Resize(thumbnail, im)
	return im


def meanshiftUsingPCA(path):
	# Load original image given the image path
	im = cv.LoadImageM(path)
	#convert image to YUV color space
	cv.CvtColor(im,im,cv.CV_BGR2YCrCb)
	# Load bank of filters
	filterBank = lmfilters.loadLMFilters()
	# Resize image to decrease dimensions during clustering
	resize_factor = 1
	thumbnail = cv.CreateMat(im.height / resize_factor, im.width / resize_factor, cv.CV_8UC3)
	cv.Resize(im, thumbnail)
	# now work with resized thumbnail image
	response = np.zeros(shape=((thumbnail.height)*(thumbnail.width),51), dtype=float)
	for f in xrange(0,48):
		filter = filterBank[f]
		# Resize the filter with the same factor for the resized image
		dst = cv.CreateImage(cv.GetSize(thumbnail), cv.IPL_DEPTH_32F, 3)
		resizedFilter = cv.CreateMat(filter.height / resize_factor, filter.width / resize_factor, filter.type)
		cv.Resize(filter, resizedFilter)
		# Apply the current filter
		cv.Filter2D(thumbnail,dst,resizedFilter)
		for j in xrange(0,thumbnail.height):
			for i in xrange(0,thumbnail.width):
				# Select the max. along the three channels
				maxRes = max(dst[j,i])
				if math.isnan(maxRes):
					maxRes = 0.0
				if maxRes > response[thumbnail.width*j+i,f]:
					# Store the max. response for the given feature index
					response[thumbnail.width*j+i,f] = maxRes

	#YUV features
	count = 0
	for j in xrange(0,thumbnail.height):
		for i in xrange(0,thumbnail.width):
			response[count,48] = thumbnail[j,i][0]
 			response[count,49] = thumbnail[j,i][1]
			response[count,50] = thumbnail[j,i][2]
            		count+=1

	#get the first 4 primary components using pca
	pca = PCA(response)
	pcaResponse = zeros([thumbnail.height*thumbnail.width,4])

	for i in xrange(0,thumbnail.height*thumbnail.width):
		pcaResponse[i] = pca.getPCA(response[i],4)
	
	# Create new mean shift instance
	ms = MeanShift(bandwidth=10,bin_seeding=True)
	# Apply the mean shift clustering algorithm
	ms.fit(pcaResponse)
	labels = ms.labels_
	n_clusters_ = np.unique(labels)
	print "Number of clusters: ", len(n_clusters_)
	repaintImage(thumbnail,labels)
	cv.Resize(thumbnail, im)
	return im

#name = "../test images/general/58060.jpg"
name = "../test images/single object/189080.jpg"

#im = meanshiftUsingIntensity(name)
#im = meanshiftUsingILM(name)
im = meanshiftUsingPCA(name)
#im = meanshiftUsingIntensityAndLocation(name)
#im = meanshiftUsingRGB(name)
#im = meanshiftUsingYUV(name)

cv.ShowImage("win1",im)
cv.WaitKey(0)
