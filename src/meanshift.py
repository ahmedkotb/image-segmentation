#!/usr/bin/env python

import cv
from numpy import *
import pymeanshift as pms

def meanshiftUsingIntensity(path):
    im = cv.LoadImageM(path,cv.CV_LOAD_IMAGE_GRAYSCALE)
    (segmentedImage, labelsImage, numberRegions) = pms.segmentMeanShift(im)
    print "number of region" , numberRegions
    return segmentedImage

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
im = meanshiftUsingIntensity(name)
#im = meanshiftUsingRGB(name)
#im = meanshiftUsingYUV(name)

cv.ShowImage("win1",im)
cv.WaitKey(0)
