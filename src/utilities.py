import glob
import numpy as np
from tqdm import tqdm, trange
import argparse
import json
import math
from myParser import parameter_parser
import random
import tensorflow as tf
import networkx as nx
from hausdorff import hausdorff_distance


def find_loss(prediction, target):
    """
    Calculating the squared loss on the normalized GED.
    """
    prediction = prediction
    target = target
    score = (prediction-target)**2
    return score


def process(path):
    data = json.load(open(path))
    return data

class data2:
    """
    Class for data gathering Not much use can be changed to proper format later
    """
    def __init__(self):
        self.args = parameter_parser()
        print("\nEnumerating unique labels.\n")
        self.training_graphs = glob.glob(self.args.training_graphs + "*.json")
        self.testing_graphs = glob.glob(self.args.testing_graphs + "*.json")
        self.graph_pairs = self.training_graphs + self.testing_graphs
        self.global_labels = set()
        for self.graph_pair in tqdm(self.graph_pairs):
            self.data = process(self.graph_pair)
            self.global_labels = self.global_labels.union(set(self.data["labels_1"]))
            self.global_labels = self.global_labels.union(set(self.data["labels_2"]))
        self.global_labels = list(self.global_labels)
        self.global_labels = {val:index  for index, val in enumerate(self.global_labels)}
        self.number_of_labels = len(self.global_labels)
        #print("Number of labels:")
        #print(self.number_of_labels)
        #print(self.global_labels)
    
    def getlabels(self):
        return self.global_labels
    def getnumlabels(self):
        return self.number_of_labels
    def gettrain(self):
        return self.training_graphs
    def gettest(self):
        return self.testing_graphs
    def create_batches(self):
        #random.shuffle(self.training_graphs)
        batches = []
        for graph in range(0, len(self.training_graphs), 32):
            batches.append(self.training_graphs[graph:graph+32])
        return batches

def transfomer(graph_1, graph_2):

    graph_1.remove_nodes_from(nx.isolates(graph_1))
    graph_2.remove_nodes_from(nx.isolates(graph_2)) 
    edges_1 = [[edge[0], edge[1]] for edge in graph_1.edges()]
    nodes_2 = graph_2.nodes()
    random.shuffle(list(nodes_2))
    mapper = {node:i for i, node in enumerate(nodes_2)}
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
    return max2

def calculate_ged(G1, G2):
        #print(type(G1))
        #print(G1.shape)
        #print(G1)
        val_diff = G1 - G2
        ged = np.sum(abs(val_diff))/2
        
        return ged

def convert_to_keras(data, global_labels):
        transformed_data = dict()

        """ 
        Converting the edge list to adjacency matrix
        edges_1 and edges_2 are edge list to adjacency matrix
        """
        s = (9,9)
        edges_1 = np.ones(s)
        np.fill_diagonal(edges_1, 0)
        edges_2 = np.ones(s)
        np.fill_diagonal(edges_2, 0)
        
        adj_matrix_1 = np.array(data["adj_matrix_1"])
        adj_matrix_2 = np.array(data["adj_matrix_2"])
        #print("edges_2:  ", edges_2)

        """ 
        Feature transforming 
        """
        features_1 = np.array(data["features_1"])
        features_2 = np.array(data["features_2"])

        norm_ged = hausdorff_distance(features_1, features_2, distance='manhattan')
        features_1 = tf.convert_to_tensor(features_1, dtype=tf.float32)
        features_2 = tf.convert_to_tensor(features_2, dtype=tf.float32)
        transformed_data["edge_index_1"] = edges_1
        transformed_data["edge_index_2"] = edges_2
        transformed_data["features_1"] = features_1
        transformed_data["features_2"] = features_2
        #norm_ged = hausdorff_distance(features_1, features_2, distance='manhattan')
        
        #calculate_ged(adj_matrix_1, adj_matrix_2)/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
        #norm_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
        
        transformed_data["target"] = tf.reshape(tf.convert_to_tensor(np.exp(-norm_ged).reshape(1, 1)),-1)
        #print(transformed_data["target"].shape)
        return transformed_data    


def convert_to_keras2(data, global_labels):
        transformed_data = dict()

        """ 
        Converting the edge list to adjacency matrix
        edges_1 and edges_2 are edge list to adjacency matrix
        """

        edges_1 = data["graph_1"] + [[y, x] for x, y in data["graph_1"]]
        
        #print("Edges_1: ", edges_1)
        size = max(max(edges_1))+1 
        #print("size ", size)
        r = [[0 for i in range(size)] for j in range(size)]
        for row,col in edges_1:
            r[row][col] = 1
        r=np.array(r)
        edges_1 = r
        edges_2 = data["graph_2"] + [[y, x] for x, y in data["graph_2"]]
        #print("Edges_2: ", edges_2)
        size = max(max(edges_2))+1
        r = [[0 for i in range(size)] for j in range(size)]
        for row,col in edges_2:
            r[row][col] = 1
        r=np.array(r)
        edges_2 = r

        """ 
        Feature transforming 
        """
        features_1, features_2 = [], []
        for n in data["labels_1"]:
            features_1.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])
        for n in data["labels_2"]:
            features_2.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])
        print("results:: ", features_1)
        features_1 = tf.convert_to_tensor(np.array(features_1), dtype=tf.float32)
        features_2 = tf.convert_to_tensor(np.array(features_2), dtype=tf.float32)
        transformed_data["edge_index_1"] = edges_1
        transformed_data["edge_index_2"] = edges_2
        transformed_data["features_1"] = features_1
        transformed_data["features_2"] = features_2
        norm_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
        #print(norm_ged.shape)
        transformed_data["target"] = tf.reshape(tf.convert_to_tensor(np.exp(-norm_ged).reshape(1, 1)),-1)
        #print(transformed_data["target"].shape)
        return transformed_data    

x = data2()