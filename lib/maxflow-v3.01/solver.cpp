#include <stdio.h>
#include "graph.h"
#include <iostream>
#include <fstream>
#include <string>
#include <map>
using namespace std;


map<string,int> nodes;
map<int,string> inverse_nodes;

int main(int argc,char* argv[]){
    if (argc < 2){
        cout << "missing file path";
        return 1;
    }
    ifstream ifs (argv[1]);
    int index = 0;
    string node = ""; 
    int number_of_nodes = 0;
    int number_of_edges = 0;
    ifs >> number_of_nodes >> number_of_edges;
	typedef Graph<double,double,double> GraphType;
	GraphType *g = new GraphType(number_of_nodes,number_of_edges); 
    for (int i=0;i<number_of_nodes;i++)
        g->add_node();
    ifs >> node;
    while (node != "=="){
        int node_index = 0;
        double w1,w2;
        ifs >> w1 >> w2;
        if (nodes.count(node) == 0){
            nodes[node] = index;
            inverse_nodes[index] = node;
            index++;
        }
        node_index = nodes[node];
        g -> add_tweights( node_index, w1, w2 );
        ifs >> node;
    }
    while (ifs >> node){
        string n1,n2;
        double w;
        n1 = node;
        ifs >> n2;
        ifs >> w;
        if (nodes.count(n1) == 0){
            nodes[n1] = index;
            inverse_nodes[index] = n1;
            index++;
        }
        if (nodes.count(n2) == 0){
            nodes[n2] = index;
            inverse_nodes[index] = n2;
            index++;
        }
        g -> add_edge(nodes[n1] , nodes[n2], w,0);
    }
    ifs.close();

	double flow = g -> maxflow();

	//printf("Flow = %d\n", flow);
	//printf("Minimum cut:\n");
    for (int i=0;i<number_of_nodes;i++){
        if (g->what_segment(i) == GraphType::SOURCE)
            cout << inverse_nodes[i] << " " << "src\n";
        else
            cout << inverse_nodes[i] << " " << "snk\n";
    }

    delete g;
    return 0;
}
