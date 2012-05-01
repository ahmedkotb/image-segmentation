#!/usr/bin/env python

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
    mat = cv.CreateMat(im.height,im.width,cv.CV_32FC3)
    for i in xrange(0,im.height):
        for j in xrange(0,im.width):
            value = (im[i,j],i,j)
            mat[i,j] = value

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

name = "../test images/single object/189080.jpg"
#name = "../test images/single object/69015.jpg"

im = meanshiftUsingIntensity(name)
#im = meanshiftUsingIntensityAndLocation(name)
#im = meanshiftUsingRGB(name)
#im = meanshiftUsingYUV(name)

cv.ShowImage("win1",im)
cv.WaitKey(0)
