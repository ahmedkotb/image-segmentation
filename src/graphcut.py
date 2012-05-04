import cv
import math
import time

from pygraph.classes.digraph import *
from pygraph.algorithms.minmax import *
from pygraph.algorithms.searching import *
from numpy import *
#--------------------------------------------------------------------------

dx = [-1,0,1,1,1,0,-1,-1]
dy = [-1,-1,-1,0,1,1,1,0]
gr = None
src = None
dest = None
pixel_to_node = None
src_hist = None
dest_hist = None
SIGMA = 2.0
LAMBDA = 60
INFINITY = 1000000000
FOREGROUND_COLOR = 0
BACKGROUND_COLOR = 255
EPS = 0.0000001
RGB_DISC = 1
INT_DISC = 1
XLOC_DISC = 5
YLOC_DISC = 5
#--------------------------------------------------------------------------

def isValid(row, col, maxRow, maxCol):
    return (row >= 0 and row < maxRow and col >= 0 and col < maxCol)
#--------------------------------------------------------------------------

def affinity(v1, v2, feature_type):
    dist = 0.0
    if feature_type == "INTENSITY":
        dist = abs(v1[0] - v2[0])
    elif feature_type == "INTENSITY+LOC" or feature_type == "RGB" or feature_type == "YUV":
        dx = v1[0] - v2[0]
        dy = v1[1] - v2[1]
        dz = v1[2] - v2[2]
        dist = sqrt(dx * dx + dy * dy + dz * dz)

    sim = math.exp((-1/(2*SIGMA*SIGMA)) * dist * dist)
#    sim = math.exp((-1/(2*SIGMA*SIGMA)) * dist)
    return sim
#--------------------------------------------------------------------------

def constructHist(fv_grid, sample, feature_type):
    hist = None
    slen = len(sample)
    n = len(fv_grid[0][0])

    if feature_type == 'INTENSITY':
        hist = zeros((256))
        for s in range(0, slen):
            for i in range(0, n):
                hist[fv_grid[sample[s][0],sample[s][1],i]] += 1.0 / slen
    elif feature_type == 'RGB' or feature_type == 'YUV':
        dim = 256/RGB_DISC
        hist = zeros ((dim, dim, dim))
        for s in range(0, slen):
            y = sample[s][0]
            x = sample[s][1]
            hist[fv_grid[y, x, 0]/RGB_DISC, fv_grid[y, x, 1]/RGB_DISC, fv_grid[y, x, 2]/RGB_DISC] += 1.0 / slen
    elif feature_type == 'INTENSITY+LOC':
        dim1 = 256/INT_DISC + 1
        dim2 = len(fv_grid)/YLOC_DISC + 1
        dim3 = len(fv_grid[0])/XLOC_DISC + 1
        hist = zeros((dim1, dim2, dim3))
        for s in range(0, slen):
            y = sample[s][0]
            x = sample[s][1]
            hist[fv_grid[y, x, 0]/INT_DISC, y/YLOC_DISC, x/XLOC_DISC] += 1.0 / slen
                
    return hist    
#--------------------------------------------------------------------------

def regionalCost(feature_vector, hist, feature_type):
    cost = 0
    if feature_type == 'INTENSITY':
        if hist[feature_vector[0]] < EPS:
            cost = INFINITY
        else:
            cost = LAMBDA * -1 * math.log(hist[feature_vector[0]])
    elif feature_type == 'RGB' or feature_type == 'YUV':
        if hist[feature_vector[0]/RGB_DISC, feature_vector[1]/RGB_DISC, feature_vector[2]/RGB_DISC] < EPS:
            cost = INFINITY
        else:
            cost = LAMBDA * -1 * math.log(hist[feature_vector[0]/RGB_DISC, feature_vector[1]/RGB_DISC, feature_vector[2]/RGB_DISC])
    elif feature_type == 'INTENSITY+LOC':
        if hist[feature_vector[0]/INT_DISC, feature_vector[1]/YLOC_DISC, feature_vector[2]/XLOC_DISC] < EPS:
            cost = INFINITY
        else:
            cost = LAMBDA * -1 * math.log(hist[feature_vector[0]/INT_DISC, feature_vector[1]/YLOC_DISC, feature_vector[2]/XLOC_DISC])
     
    return cost
#--------------------------------------------------------------------------

def constructTLinks(fv_grid, gr, pixel_to_node, feature_type, obj_sample, bk_sample):
    obj_index = 0
    bk_index = 0
    objlen = len(obj_sample)
    bklen = len(bk_sample)

    height = len(fv_grid)
    width = len(fv_grid[0])
    for y in range(0,height):
        for x in range(0,width):
            scost = 0
            tcost = 0
            crnt = pixel_to_node[y][x]
            if(obj_index < objlen and y == obj_sample[obj_index][0] and x == obj_sample[obj_index][1]):
                scost = INFINITY
                tcost = 0
                obj_index += 1
            elif(bk_index < bklen and y == bk_sample[bk_index][0] and x == bk_sample[bk_index][1]):
                scost = 0
                tcost = INFINITY
                bk_index += 1
            else:
                scost = regionalCost(fv_grid[y][x], dest_hist, feature_type)
                tcost = regionalCost(fv_grid[y][x], src_hist, feature_type)
            gr.add_edge((src, crnt), scost)
            gr.add_edge((crnt, dest), tcost)

#--------------------------------------------------------------------------

def constructNLinkes(feature_vector, gr, pixel_to_node, feature_type):
    height = len(feature_vector)
    width = len(feature_vector[0])
    for y in range(0, height):
        for x in range(0,width):
            crnt = pixel_to_node[y][x]
            for i in range(0,8):
                yy = y + dy[i]
                xx = x + dx[i]
                if isValid(yy, xx, height, width):
                    neighbor = pixel_to_node[yy][xx]
                    cost = affinity(feature_vector[y,x], feature_vector[yy,xx], feature_type)
                    gr.add_edge((crnt, neighbor), cost)
#--------------------------------------------------------------------------

def constructEdges(feature_vector, gr, pixel_to_node, feature_type, obj_sample, bk_sample):
    constructNLinkes(feature_vector, gr, pixel_to_node, feature_type)
    constructTLinks(feature_vector, gr, pixel_to_node, feature_type, obj_sample, bk_sample)
#--------------------------------------------------------------------------

def constructNodes(fv_grid, gr, feature_type):
    global src
    global dest
    global pixel_to_node
    height = len(fv_grid)
    width = len(fv_grid[0])
    src = 'n0'
    dest = 'n' + str(width * height + 1)
    print 'src =',src
    print 'dest =', dest
    gr.add_nodes([src])
    pixel_to_node = []
    index = 1

    for y in range(0, height):
        pixel_to_node.append([])
        for _ in range(0, width):
            node_name = ('n'+str(index))
            index += 1
            pixel_to_node[y].append(node_name)
            gr.add_nodes([node_name])

    gr.add_nodes([dest])
#--------------------------------------------------------------------------

def constructFeatureVector(image, feature_type):
    v = None
    if feature_type == "INTENSITY":
        v = zeros((image.height,image.width,1))
    elif feature_type == "INTENSITY+LOC":
        v = zeros((image.height,image.width,3))
    elif feature_type == "RGB":
        v = zeros((image.height,image.width,3))
    elif feature_type == "YUV":
        v = zeros((image.height,image.width,3))

    for y in range(0,image.height):
        for x in range(0, image.width):
            if feature_type == "INTENSITY":
                v[y,x,0] = image[y,x]
            elif feature_type == "INTENSITY+LOC":
                v[y,x,0] = image[y,x]
                v[y,x,1] = y
                v[y,x,2] = x
            elif feature_type == "RGB" or feature_type == "YUV":
                v[y,x,0] = image[y,x][0]
                v[y,x,1] = image[y,x][1]
                v[y,x,2] = image[y,x][2]
            # add more features here
    return v
#--------------------------------------------------------------------------

def constructGraph(img, feature_type, obj_sample, bk_sample):
    global gr
    global src_hist
    global dest_hist
    gr = digraph()
    feature_vectors = constructFeatureVector(img, feature_type)
    src_hist = constructHist(feature_vectors, obj_sample, feature_type)
    dest_hist = constructHist(feature_vectors, bk_sample, feature_type)
    constructNodes(feature_vectors, gr, feature_type)
    constructEdges(feature_vectors, gr, pixel_to_node, feature_type, obj_sample, bk_sample)
#--------------------------------------------------------------------------

def construct_graph(image, feature_type):
    global gr
    gr = digraph()
    feature_vector = constructFeatureVector(image, feature_type)
    constructNodes(feature_vector, gr, feature_type)
    constructEdges(feature_vector, gr, pixel_to_node, feature_type)
#--------------------------------------------------------------------------

def graphcut(image_name, feature_type, obj_sample, bk_sample):
    start = time.time()
    img = None
    if feature_type == "INTENSITY":
        img = cv.LoadImageM(image_name,cv.CV_LOAD_IMAGE_GRAYSCALE)
    elif feature_type == "INTENSITY+LOC":
        img = cv.LoadImageM(image_name,cv.CV_LOAD_IMAGE_GRAYSCALE)
    elif feature_type == "RGB":
        img = cv.LoadImageM(image_name,cv.CV_LOAD_IMAGE_COLOR)
    elif feature_type == "YUV":
        img = cv.LoadImageM(image_name,cv.CV_LOAD_IMAGE_COLOR)
        cv.CvtColor(img,img,cv.CV_BGR2YCrCb)
        
    constructGraph(img, feature_type, obj_sample, bk_sample)

    print "start max_flow"
    _, cut = maximum_flow(gr, src, dest);
    print cut
    gray = cv.LoadImageM(image_name,cv.CV_LOAD_IMAGE_GRAYSCALE)
    obj = cut[src]
    bk = cut[dest]
    for y in range(0, img.height):
        for x in range(0,img.width):
            if(cut[pixel_to_node[y][x]] == obj):
                gray[y,x] = FOREGROUND_COLOR
            elif(cut[pixel_to_node[y][x]] == bk):
                gray[y,x] = BACKGROUND_COLOR


    end = time.time()

    print "time",end - start,"seconds"

    return gray
#--------------------------------------------------------------------------

def segmentUsingGraphcut(img_name, feature_type, obj_sample, bk_sample):
    print "image name =", img_name
    img = graphcut(img_name, feature_type, obj_sample, bk_sample)
    cv.ShowImage("img",img)
    cv.WaitKey(0)
    cv.SaveImage("output.jpg", img)
#--------------------------------------------------------------------------
