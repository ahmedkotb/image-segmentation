#!/usr/bin/env python
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

def meanshiftUsingILM(path):
	im = cv.LoadImageM(path)
	cv.ShowImage("win2",im)
	#array of filter kernels
	filterBank = lmfilters.loadLMFilters()

	def getFilterTypeIndex(index):
		if index >=0 and index <= 17:
			return 0
		elif index >= 18 and index <= 35:
			return 1
		elif index >= 36 and index <= 45:
			return 2
		else:
			return 3

	for f in xrange(0,48):
		filter = filterBank[f]
		dst = cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_32F, 3)
		name = "result"+str(f)+".jpg"
		response = cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_32F, 4)
		cv.Filter2D(im,dst,filter)
		featureIndex = getFilterTypeIndex(f)
		if f == 5:
			cv.ShowImage("image",dst)
		#cc8u = cv.CreateMat(cv.GetSize(dst), cv.CV_8U, 1)
		#ccmin,ccmax,minij,maxij = cv.MinMaxLoc(dst)
		#ccscale, ccshift = 255.0/(ccmax-ccmin), -ccmin
		#CvtScale(dst, cc8u, ccscale, ccshift)
		#SaveImage(name, cc8u)
		#cv.SetZero(response)
		for j in xrange(0,im.height):
			for i in xrange(0,im.width):
				#take the maximum response from the 3 channels
				maxRes = max(dst[j,i])
				if math.isnan(maxRes):
					maxRes = 0.0
				#take the maximun over this feature
				if maxRes > response[j,i][featureIndex]:
					features = list(response[j,i])
					features[featureIndex] = maxRes
					response[j,i] = tuple(features)

	clusters = {}
    	(segmentedImage, labelsImage, numberRegions) = pms.segmentMeanShift(response)
	for i in xrange(0,labelsImage.height):
		for j in xrange(0,labelsImage.width):
			v = labelsImage[i,j]
			if v in clusters:
				clusters[v].append(im[i,j])
			else:
 				clusters[v] =  [im[i,j]]

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

	for i in xrange(0,im.height):
		for j in xrange(0,im.width):
 			lbl = labelsImage[i,j]
			im[i,j] = means[lbl]

	print "number of region" , numberRegions
	return im




name = "../test images/general/58060.jpg"
#name = "../test images/single object/69015.jpg"

#im = meanshiftUsingIntensity(name)
im = meanshiftUsingILM(name)
#im = meanshiftUsingIntensityAndLocation(name)
#im = meanshiftUsingRGB(name)
#im = meanshiftUsingYUV(name)

cv.ShowImage("win1",im)
cv.WaitKey(0)
