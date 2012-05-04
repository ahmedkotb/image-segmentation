#!/usr/bin/env python

from pygraph.classes.digraph import *

def write(graph,filename,src,dst):
    f = open(filename,"w")
    #first write edges from src to pixel and from
    #pixel to sink
    f.write(str(len(graph.nodes())-2) + " " + str(len(graph.edges())) + "\n")
    nodes = graph.nodes()
    for node in nodes:
        if node == src or node == dst:
            continue
        s = node + " "
        s += str(graph.edge_weight((src,node))) + " "
        s += str(graph.edge_weight((node,dst)))
        f.write(s + "\n")
    f.write("==\n")
    #second write edges between normal pixels
    for edge in graph.edges():
        if edge[0] == src or edge[1] == src:
            continue
        if edge[0] == dst or edge[1] == dst:
            continue
        s = str(edge[0])+ " "
        s += str(edge[1]) + " "
        s += str(graph.edge_weight(edge))
        f.write(s+ "\n")
    f.close()


def parse(filename,src,dst):
    cut = {}
    cut[src] = "src"
    cut[dst] = "snk"
    f = open(filename,"r")
    for line in f:
        data = line.split()
        cut[data[0]] = data[1]
    f.close()
    return cut

