# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 12:07:08 2021

@author: 91876
"""

import networkx as nx
import random
import json
from tqdm import tqdm
import os.path
import numpy as np

# Max index of graph to generate. Number og graphs generated = 1001 - index
MAX_GRAPHS = 1001

def transfomer(graph_1, graph_2):

    graph_1.remove_nodes_from(nx.isolates(graph_1))
    graph_2.remove_nodes_from(nx.isolates(graph_2)) 
    edges_1 = [[edge[0], edge[1]] for edge in graph_1.edges()]
    nodes_2 = graph_2.nodes()
    random.shuffle(list(nodes_2))
    mapper = {node:i for i, node in enumerate(nodes_2)}
    print("mapper: ", mapper)
    edges_2 = [[mapper[edge[0]], mapper[edge[1]]] for edge in graph_2.edges()]

    graph_1 = nx.from_edgelist(edges_1)
    graph_2 = nx.from_edgelist(edges_2)
    graph_1.remove_nodes_from(nx.isolates(graph_1))
    graph_2.remove_nodes_from(nx.isolates(graph_2))
    edges_1 = [[edge[0], edge[1]] for edge in graph_1.edges()]
    edges_2 = [[edge[0], edge[1]] for edge in graph_2.edges()]
    data = dict()
    data["graph_1"] = edges_1
    data["graph_2"] = edges_2
    data["labels_1"] = [str(graph_1.degree(node)) for node in graph_1]
    data["labels_2"] = [str(graph_2.degree(node))  for node in graph_2]
    nx.set_node_attributes(graph_1, 'Labels', 1)
    nx.set_node_attributes(graph_2, 'Labels', 2)
    print(nx.get_node_attributes(graph_1, "Labels"))
    for x in range(0, len(data["labels_1"])):
        graph_1.nodes[x]["Labels"] = data["labels_1"][x]
    for x in range(0, len(data["labels_2"])):
        graph_2.nodes[x]["Labels"] = data["labels_2"][x]

    print(nx.get_node_attributes(graph_1, "Labels"))
    print(nx.get_node_attributes(graph_2, "Labels"))
    max2=0
    # Finding approximate GED
    for v in nx.optimize_graph_edit_distance(graph_1, graph_2):
        max2 = v
        break
    data["ged"] = max2
    print("Graph Edit distance is:")
    print( data["ged"] )
    return 1005


def substitution_cost(edge1, edge2):
    return abs(edge1['weight'] - edge2['weight'])

def match_edge(edge1, edge2):
    return (edge1['weight'] == edge2['weight'])

def calculate_ged(G1, G2):
        #print(type(G1))
        #print(G1.shape)
        #print(G1)
        val_diff = G1 - G2
        ged = np.sum(abs(val_diff))/2
        
        return ged
    
# Starting index (Used to keep the initialial dataset intact. (Overlapping the original dataset may give label errors.
error = 0
path = "C:/Users/91876/Downloads/SimGNN-main/dataset/A01/Training/A01T_1/24.json"
data = json.load(open(path))
adj_matrix_1 = np.round_(np.array(data["adj_matrix_1"]) * 100000).astype(int)
adj_matrix_2 = np.round_(np.array(data["adj_matrix_2"]) * 100000).astype(int)
temp = nx.MultiGraph()
graph_1 = nx.from_numpy_matrix(adj_matrix_1)
graph_2 = nx.from_numpy_matrix(adj_matrix_2)
#index = transfomer(graph_1, graph_2)
print(graph_1[3][4]['weight'])

max2=0
# Finding approximate GED
for v in nx.optimize_graph_edit_distance(graph_1, graph_2,edge_match= match_edge, edge_subst_cost=substitution_cost):
    max2 = v
    break
data["ged"] = max2
print("Graph Edit distance is:")
print( data["ged"] )

print("ged from earlier method ", calculate_ged(adj_matrix_1, adj_matrix_2))









