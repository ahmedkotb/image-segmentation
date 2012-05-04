#from pygraph.classes.graph import *
#from pygraph.algorithms.searching import *

#mypygraph =  __import__('pygraph.classes.graph')
import cv
import math
import time

from pygraph.classes.digraph import *
from pygraph.algorithms.minmax import *
from pygraph.algorithms.searching import *
from numpy import *

dx = [-1,0,1,1,1,0,-1,-1]
dy = [-1,-1,-1,0,1,1,1,0]
gr = None
src = None
dest = None
pixel_to_node = None
src_hist = None
dest_hist = None
SIGMA = 2
LAMDA = 60
INFINITY = 1000000000
#--------------------------------------------------------------------------

def isValid(row, col, maxRow, maxCol):
    return (row >= 0 and row < maxRow and col >= 0 and col < maxCol)
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# TODO remove this method .. unused
def affinityUsingIntensity(int1, int2):
    dist = abs(int1-int2)
    sim = math.exp((-1/(2*SIGMA*SIGMA)) * dist * dist)
    return sim

def affinity(v1, v2, feature_type):
    dist = 0.0
    if feature_type == "INTENSITY":
        dist = abs(v1[0] - v2[0])
    elif feature_type == "INTENSITY+LOC":
        dx = v1[0] - v2[0]
        dy = v1[1] - v2[1]
        dz = v1[2] - v2[2]
        dist = sqrt(dx * dx + dy * dy + dz * dz)
    elif feature_type == "RGB":
        dx = v1[0] - v2[0]
        dy = v1[1] - v2[1]
        dz = v1[2] - v2[2]
        dist = sqrt(dx * dx + dy * dy + dz * dz)

    sim = math.exp((-1/(2*SIGMA*SIGMA)) * dist * dist)
#    sim = math.exp((-1/(2*SIGMA*SIGMA)) * dist)
    return sim
#--------------------------------------------------------------------------

def constructHist(fv_grid, sample, type):
    hist = None
    slen = len(sample)
    n = len(fv_grid[0][0])
    
    if type == 'INTENSITY':
        hist = zeros((256))
    
        for s in range(0, slen):
            for i in range(0, n):
                hist[fv_grid[sample[s][0],sample[s][1],i]] += 1.0 / slen
#        for y in range(0, len(fv_grid)):
#            for x in range(0, len(fv_grid[0])):
#                for i in range(0, n):
#                    hist[fv_grid[y,x,i]] += 1

                
#    print hist
    return hist    
#--------------------------------------------------------------------------

def regionalCost(feature_vector, hist, feature_type):
    cost = 0
    if feature_type == 'INTENSITY':
        cost = LAMBDA * hist[feature_vector[0]]
        
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

def constructNodes(fv_map, gr, feature_type):
    global src
    global dest
    global pixel_to_node
    height = len(fv_map)
    width = len(fv_map[0])
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

def construct_feature_vector(image, feature_type):
    v = None
    if feature_type == "INTENSITY":
        v = zeros((image.height,image.width,1))
    elif feature_type == "INTENSITY+LOC":
        v = zeros((image.height,image.width,3))
    elif feature_type == "RGB":
        v = zeros((image.height,image.width,3))
    # //XXX add more features here
    
    for y in range(0,image.height):
        for x in range(0, image.width):
            if feature_type == "INTENSITY":
                v[y,x,0] = image[y,x]
            elif feature_type == "INTENSITY+LOC":
                v[y,x,0] = image[y,x]
                v[y,x,1] = y
                v[y,x,2] = x
            elif feature_type == "RGB":
                v[y,x,0] = image[y,x][0]
                v[y,x,1] = image[y,x][1]
                v[y,x,2] = image[y,x][2]
            # add more features here
    return v
#--------------------------------------------------------------------------

def constructGraph(img, feature_type, obj_sample, bk_smaple):
    global gr
    global src_hist
    global dest_hist
    gr = digraph()
    feature_vectors = construct_feature_vector(img, feature_type)
    src_hist = constructHist(feature_vectors, obj_sample, feature_type)
    dest_hist = constructHist(feature_vectors, bk_smaple, feature_type)
    constructNodes(feature_vectors, gr, feature_type)
    constructEdges(feature_vectors, gr, pixel_to_node, feature_type, obj_sample, bk_sample)
    
#--------------------------------------------------------------------------

def construct_graph(image, feature_type):
    global gr
    gr = digraph()
    feature_vector = construct_feature_vector(image, feature_type)
    constructNodes(feature_vector, gr, feature_type)
    constructEdges(feature_vector, gr, pixel_to_node, feature_type)
#    constructNodes(image, gr, feature_type)
#    constructEdges(image, gr, pixel_to_node, feature_type)
#    st, pre, post = depth_first_search(gr, root='n1')
#    print 'pre ', pre

#--------------------------------------------------------------------------

def graphcutUsingIntensity(image):
    construct_graph(image, "INTENSITY")
    _, cut = maximum_flow(gr, src, dest);
    print cut
    obj = cut[src]
    bk = cut[dest]
    print obj, bk
    for y in range(0, image.height):
        for x in range(0,image.width):
            if(cut[pixel_to_node[y][x]] == obj):
                image[y,x] = 255
            elif(cut[pixel_to_node[y][x]] == bk):
                image[y,x] = 0
#--------------------------------------------------------------------------

def graphcut(image_name, feature_type, obj_sample, bk_sample):
    start = time.time()
    img = None
    if feature_type == "INTENSITY":
        img = cv.LoadImageM(image_name,cv.CV_LOAD_IMAGE_GRAYSCALE)
#        graphcutUsingIntensity(img)
    elif feature_type == "INTENSITY+LOC":
        img = cv.LoadImageM(image_name,cv.CV_LOAD_IMAGE_GRAYSCALE)
#        graphcutUsingIntensityAndLocation(img)
    elif feature_type == "RGB":
        img = cv.LoadImageM(image_name,cv.CV_LOAD_IMAGE_COLOR)
#        graphUsingRGB(img)
    elif feature_type == "YUV":
        img = cv.LoadImageM(image_name,cv.CV_LOAD_IMAGE_COLOR)
#        graphcutUsingYUV(img)
    elif feature_type == "LM":
        img = cv.LoadImageM(image_name,cv.CV_LOAD_IMAGE_COLOR)
#        graphcutUsingLM(img)
    elif feature_type == "ILM":
        img = cv.LoadImageM(image_name,cv.CV_LOAD_IMAGE_COLOR)
#        graphcutUsingILM(img)
    
    constructGraph(img, feature_type, obj_sample, bk_sample)
    
    end = time.time()

    print "time",end - start,"seconds"

    return img

#--------------------------------------------------------------------------
def segmentUsingGraphcut(img_name, feature_type, obj_sample, bk_sample):
    print "image name =", img_name
    img = None
    img = graphcut(img_name, feature_type, obj_sample, bk_sample)
    cv.ShowImage("img",img)
    cv.WaitKey(0)
    cv.SaveImage("output.jpg", img)
#name = "../test images/single object/189080.jpg"
#name = "../test images/single object/285079.jpg"
#name = "../lm_images/square2.gif"
#name = "../lm_images/square1.png"
#name = "../lm_images/grayscale1_2.png"
#print "img name = ",name
#
#img = None
#img = graphcut(name,"INTENSITY", None, None)
#
#cv.ShowImage("img",img)
#cv.WaitKey(0)
#
#cv.SaveImage("output.jpg", img)

