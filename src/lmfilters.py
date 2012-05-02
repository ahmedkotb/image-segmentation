import cv

LM_FILTERS_PATH = "../LeunMalikFilterBank.txt"

def loadLMFilters():
    filterBank = []
    for i in xrange(0,48):
        filter = cv.CreateMat(49,49,cv.CV_32FC1)
        filterBank.append(filter)
    f = open(LM_FILTERS_PATH,"r")
    count = 0
    for line in f:
        line = line.strip()
        if line == "":
            continue
        values = line.split()
        for i in xrange(0,49):
            filter = filterBank[count/49]
            filter[count%49,0] = float(values[i])
        count+=1
    f.close()
    return filterBank

