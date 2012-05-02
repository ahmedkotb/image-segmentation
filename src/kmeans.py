import cv
import cv2
import math
import time
import lmfilters
from pca import *
from numpy import *


def getClusters(imgCol, samples, labels):
    clusters = {}
    for i in xrange(0,samples.rows):
        v = samples[i,0]
        lbl = labels[i,0]
        try:
            clusters[lbl].append(imgCol[i,0])
        except KeyError:
            clusters[lbl] = [ imgCol[i,0] ]
    return clusters


def kmeansUsingIntensity(im,k,iterations,epsilon):
    #create the samples and labels vector
    col = cv.Reshape(im, 1,im.width*im.height)
    samples = cv.CreateMat(col.height, 1, cv.CV_32FC1)
    cv.Scale(col,samples)

    labels = cv.CreateMat(col.height, 1, cv.CV_32SC1)
    crit = (cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, iterations, epsilon)
    cv.KMeans2(samples, k, labels, crit)

    #calculate the means
    clusters = getClusters(col,samples,labels)

    means = {}
    for c in clusters:
        means[c] = sum(clusters[c])/len(clusters[c])

    for m in means:
        print m,means[m],len(clusters[m])

    #apply clustering to the image
    for i in xrange(0,col.rows):
        lbl = labels[i,0]
        col[i,0] = means[lbl]

def kmeansUsingIntensityAndLocation(im,k,iterations,epsilon):
    #create the samples and labels vector
    col = cv.Reshape(im, 1,im.width*im.height)
    samples = cv.CreateMat(col.height, 3, cv.CV_32FC1)
    count = 0
    for j in xrange(0,im.height):
        for i in xrange(0,im.width):
            samples[count,0] = im[j,i]
            samples[count,1] = i
            samples[count,2] = j
            count+=1

    labels = cv.CreateMat(col.height, 1, cv.CV_32SC1)
    crit = (cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, iterations, epsilon)
    cv.KMeans2(samples, k, labels, crit)

    clusters = getClusters(col,samples,labels)

    #calculate the means
    means = {}
    for c in clusters:
        means[c] = sum(clusters[c])/len(clusters[c])

    for m in means:
        print m,means[m],len(clusters[m])
    #apply clustering to the image
    for i in xrange(0,col.rows):
        lbl = labels[i,0]
        col[i,0] = means[lbl]

def kmeansUsingRGB(im,k,iterations,epsilon):
    #create the samples and labels vector
    col = cv.Reshape(im, 3,im.width*im.height)
    samples = cv.CreateMat(col.height, 1, cv.CV_32FC3)
    cv.Scale(col,samples)

    labels = cv.CreateMat(col.height, 1, cv.CV_32SC1)
    crit = (cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, iterations, epsilon)
    cv.KMeans2(samples, k, labels, crit)
    #calculate the means
    clusters = getClusters(col,samples,labels)

    means = {}
    for c in clusters:
        means[c] = [0.0,0.0,0.0]
        for v in clusters[c]:
            means[c][0] += v[0]
            means[c][1] += v[1]
            means[c][2] += v[2]
        means[c][0] /= len(clusters[c])
        means[c][1] /= len(clusters[c])
        means[c][2] /= len(clusters[c])

    for m in means:
        print m,means[m],len(clusters[m])
    #apply clustering to the image
    for i in xrange(0,col.rows):
        lbl = labels[i,0]
        col[i,0] = means[lbl]

def kmeansUsingYUV(im,k,iterations,epsilon):
    cv.CvtColor(im,im,cv.CV_BGR2YCrCb)
    kmeansUsingRGB(im,k,iterations,epsilon)


def kmeansUsingLM(im,k,iterations,epsilon):
    #array of filter kernels
    filterBank = lmfilters.loadLMFilters()

    #create the samples and labels vector
    col = cv.Reshape(im, 3,im.width*im.height)
    samples = cv.CreateMat(col.height, 48, cv.CV_32FC1)

    for f in xrange(0,48):
        filter = filterBank[f]
        dst = cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_32F, 3)
        cv.Filter2D(im,dst,filter)
        count = 0
        for j in xrange(0,im.height):
            for i in xrange(0,im.width):
                #take the maximum response from the 3 channels
                maxRes = max(dst[j,i])
                if math.isnan(maxRes):
                    maxRes = 0.0
                samples[count,f] = maxRes
                count+=1

    labels = cv.CreateMat(col.height, 1, cv.CV_32SC1)

    crit = (cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, iterations, epsilon)
    cv.KMeans2(samples, k, labels, crit)

    clusters = getClusters(col,samples,labels)


    means = {}
    for c in clusters:
        means[c] = [0.0,0.0,0.0]
        for v in clusters[c]:
            means[c][0] += v[0]
            means[c][1] += v[1]
            means[c][2] += v[2]
        means[c][0] /= len(clusters[c])
        means[c][1] /= len(clusters[c])
        means[c][2] /= len(clusters[c])

    for m in means:
        print m,means[m],len(clusters[m])

    #apply clustering to the image
    for i in xrange(0,col.rows):
        lbl = labels[i,0]
        col[i,0] = means[lbl]


def kmeansUsingILM(im,k,iterations,epsilon):
    #array of filter kernels
    filterBank = lmfilters.loadLMFilters()

    #create the samples and labels vector
    col = cv.Reshape(im, 3,im.width*im.height)
    samples = cv.CreateMat(col.height, 4, cv.CV_32FC1)
    cv.SetZero(samples)

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
        cv.Filter2D(im,dst,filter)
        count = 0
        featureIndex = getFilterTypeIndex(f)
        for j in xrange(0,im.height):
            for i in xrange(0,im.width):
                #take the maximum response from the 3 channels
                maxRes = max(dst[j,i])
                if math.isnan(maxRes):
                    maxRes = 0.0
                #take the maximun over this feature
                if maxRes > samples[count,featureIndex]:
                    samples[count,featureIndex] = maxRes
                count+=1

    labels = cv.CreateMat(col.height, 1, cv.CV_32SC1)

    crit = (cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, iterations, epsilon)
    cv.KMeans2(samples, k, labels, crit)

    clusters = getClusters(col,samples,labels)


    means = {}
    for c in clusters:
        means[c] = [0.0,0.0,0.0]
        for v in clusters[c]:
            means[c][0] += v[0]
            means[c][1] += v[1]
            means[c][2] += v[2]
        means[c][0] /= len(clusters[c])
        means[c][1] /= len(clusters[c])
        means[c][2] /= len(clusters[c])

    for m in means:
        print m,means[m],len(clusters[m])

    #apply clustering to the image
    for i in xrange(0,col.rows):
        lbl = labels[i,0]
        col[i,0] = means[lbl]

def kmeansUsingPCA(im,k,iterations,epsilon):

    #convert image to YUV color space
    cv.CvtColor(im,im,cv.CV_BGR2YCrCb)
    #array of filter kernels
    filterBank = lmfilters.loadLMFilters()

    #create the samples and labels vector
    col = cv.Reshape(im, 3,im.width*im.height)
    #samples = cv.CreateMat(col.height, 51, cv.CV_32FC1)
    samples = zeros([col.height,51])

    for f in xrange(0,48):
        filter = filterBank[f]
        dst = cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_32F, 3)
        cv.Filter2D(im,dst,filter)
        count = 0
        for j in xrange(0,im.height):
            for i in xrange(0,im.width):
                #take the maximum response from the 3 channels
                maxRes = max(dst[j,i])
                if math.isnan(maxRes):
                    maxRes = 0.0
                samples[count,f] = maxRes
                count+=1

    #YUV features
    count = 0
    for j in xrange(0,im.height):
        for i in xrange(0,im.width):
            samples[count,48] = im[j,i][0]
            samples[count,49] = im[j,i][1]
            samples[count,50] = im[j,i][2]
            count+=1

    #get the first 4 primary components using pca
    pca = PCA(samples)
    pcaSamples = zeros([col.height,4])

    for i in xrange(0,col.height):
        pcaSamples[i] = pca.getPCA(samples[i],4)

    samples = cv.fromarray(pcaSamples)
    samplesMat = cv.CreateMat(col.height, 4, cv.CV_32FC1)
    cv.Scale(samples,samplesMat)

    labels = cv.CreateMat(col.height, 1, cv.CV_32SC1)

    crit = (cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, iterations, epsilon)
    cv.KMeans2(samplesMat, k, labels, crit)

    clusters = getClusters(col,samplesMat,labels)


    means = {}
    for c in clusters:
        means[c] = [0.0,0.0,0.0]
        for v in clusters[c]:
            means[c][0] += v[0]
            means[c][1] += v[1]
            means[c][2] += v[2]
        means[c][0] /= len(clusters[c])
        means[c][1] /= len(clusters[c])
        means[c][2] /= len(clusters[c])

    for m in means:
        print m,means[m],len(clusters[m])

    #apply clustering to the image
    for i in xrange(0,col.rows):
        lbl = labels[i,0]
        col[i,0] = means[lbl]
#-------------------------------------------------------------
def kmeans(image_name,feature,k,iterations,epsilon):
    start = time.time()

    im = None
    if feature == "INTENSITY":
        im = cv.LoadImageM(name,cv.CV_LOAD_IMAGE_GRAYSCALE)
        kmeansUsingIntensity(im,k,iterations,epsilon)
    elif feature == "INTENSITY+LOC":
        im = cv.LoadImageM(name,cv.CV_LOAD_IMAGE_GRAYSCALE)
        kmeansUsingIntensityAndLocation(im,k,iterations,epsilon)
    elif feature == "RGB":
        im = cv.LoadImageM(name,cv.CV_LOAD_IMAGE_COLOR)
        kmeansUsingRGB(im,k,iterations,epsilon)
    elif feature == "YUV":
        im = cv.LoadImageM(name,cv.CV_LOAD_IMAGE_COLOR)
        kmeansUsingYUV(im,k,iterations,epsilon)
    elif feature == "LM":
        im = cv.LoadImageM(name,cv.CV_LOAD_IMAGE_COLOR)
        kmeansUsingLM(im,k,iterations,epsilon)
    elif feature == "ILM":
        im = cv.LoadImageM(name,cv.CV_LOAD_IMAGE_COLOR)
        kmeansUsingILM(im,k,iterations,epsilon)
    elif feature == "PCA":
        im = cv.LoadImageM(name,cv.CV_LOAD_IMAGE_COLOR)
        kmeansUsingPCA(im,k,iterations,epsilon)

    end = time.time()

    print "time",end - start,"seconds"

    return im

if __name__ == "__main__":
    name = "../test images/single object/189080.jpg"
    #name = "../test images/single object/69015.jpg"
    #name = "../test images/single object/291000.jpg"
    k = 3
    iterations = 100
    epsilon = 0.001

    print "img name =",name

    im = None
    im = kmeans(name,"INTENSITY",k,iterations,epsilon)
    #im = kmeans(name,"INTENSITY+LOC",k,iterations,epsilon)
    #im = kmeans(name,"RGB",k,iterations,epsilon)
    #im = kmeans(name,"YUV",k,iterations,epsilon)
    #im = kmeans(name,"LM",k,iterations,epsilon)
    #im = kmeans(name,"ILM",k,iterations,epsilon)
    #im = kmeans(name,"PCA",k,iterations,epsilon)

    cv.ShowImage("win1",im)
    cv.WaitKey(0)

