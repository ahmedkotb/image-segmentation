#from pygraph.classes.graph import *
#from pygraph.algorithms.searching import *

#mypygraph =  __import__('pygraph.classes.graph')
import cv
import math
import time

from pygraph.classes.digraph import *
from pygraph.algorithms.minmax import *
from pygraph.algorithms.searching import *

dx = [-1,0,1,1,1,0,-1,-1]
dy = [-1,-1,-1,0,1,1,1,0]
gr = None
src = None
dest = None
pixel_to_node = None
sigma = 2
#--------------------------------------------------------------------------

def isValid(row, col, maxRow, maxCol):
    return (row >= 0 and row < maxRow and col >= 0 and col < maxCol)
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

def affinityUsingIntensity(int1, int2):
    dist = abs(int1-int2)
    sim = math.exp((-1/(2*sigma*sigma)) * dist * dist)
    return sim
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

def constructTLinks(image, gr, pixel_to_node):
    src_int = 0
    dest_int = 255
    scost = 0
    tcost = 0
    inf = 999999999
    for y in range(0, image.height):
        for x in range(0,image.width):
            crnt = pixel_to_node[y][x]
            if(image[y,x] == src_int):
                scost = inf
                tcost = 0
            elif(image[y,x] == dest_int):
                scost = 0
                tcost = inf
            gr.add_edge((src, crnt), scost)
            gr.add_edge((crnt, dest), tcost)
#--------------------------------------------------------------------------

def constructNLinkes(image, gr, pixel_to_node):
    for y in range(0, image.height):
        for x in range(0,image.width):
            crnt = pixel_to_node[y][x]
            for i in range(0,8):
                yy = y + dy[i]
                xx = x + dx[i]
                if isValid(yy, xx, image.height, image.width):
                    neighbor = pixel_to_node[yy][xx]
                    cost = affinityUsingIntensity(image[y, x], image[yy, xx])
                    gr.add_edge((crnt, neighbor), cost)
#--------------------------------------------------------------------------

def constructEdges(image, gr, pixel_to_node):
    constructNLinkes(image, gr, pixel_to_node)
    constructTLinks(image, gr, pixel_to_node)
#--------------------------------------------------------------------------

def constructNodes(image, gr):
    global src
    global dest
    global pixel_to_node
    src = 'n0'
    dest = 'n'+str(image.width*image.height + 1)
    print 'src =',src
    print 'dest =', dest
    gr.add_nodes([src])
    pixel_to_node = []
    index = 1
    for y in range(0, image.height):
        pixel_to_node.append([])
        for _ in range(0,image.width):
            node_name = ('n'+str(index))
            index += 1
            pixel_to_node[y].append(node_name)
            gr.add_nodes([node_name])
    
    gr.add_nodes([dest])
#--------------------------------------------------------------------------

def construct_graph(image):
    global gr
    gr = digraph()
    constructNodes(image, gr)
    constructEdges(image, gr, pixel_to_node)
#    st, pre, post = depth_first_search(gr, root='n1')
#    print 'pre ', pre

#--------------------------------------------------------------------------

def graphcutUsingIntensity(image):
    construct_graph(image)
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

def graphcut(image_name, feature_type):
    start = time.time()
    img = None
    if feature_type == "INTENSITY":
        img = cv.LoadImageM(name,cv.CV_LOAD_IMAGE_GRAYSCALE)
        graphcutUsingIntensity(img)
    elif feature_type == "INTENSITY+LOC":
        img = cv.LoadImageM(name,cv.CV_LOAD_IMAGE_GRAYSCALE)
#        graphcutUsingIntensityAndLocation(img)
    elif feature_type == "RGB":
        img = cv.LoadImageM(name,cv.CV_LOAD_IMAGE_COLOR)
#        graphUsingRGB(img)
    elif feature_type == "YUV":
        img = cv.LoadImageM(name,cv.CV_LOAD_IMAGE_COLOR)
#        graphcutUsingYUV(img)
    elif feature_type == "LM":
        img = cv.LoadImageM(name,cv.CV_LOAD_IMAGE_COLOR)
#        graphcutUsingLM(img)
    elif feature_type == "ILM":
        img = cv.LoadImageM(name,cv.CV_LOAD_IMAGE_COLOR)
#        graphcutUsingILM(img)
    
    end = time.time()

    print "time",end - start,"seconds"

    return img

#--------------------------------------------------------------------------
#name = "../test images/single object/189080.jpg"
#name = "../test images/single object/285079.jpg"
#name = "../lm_images/square2.gif"
#name = "../lm_images/square1.png"
name = "../lm_images/grayscale1.png"
print "img name =",name

img = None
img = graphcut(name,"INTENSITY")


cv.ShowImage("img",img)
cv.WaitKey(0)

cv.SaveImage("output.jpg", img)