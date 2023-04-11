import string
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import unittest
from collections import defaultdict
import numpy as np
import pandas as pd
from copy import deepcopy


class DictGraph(dict):
    def __init__(self, vertices):
        self.vertices = vertices

    def add_edge(self, v1, v2):
        if v1 in self.vertices and v2 in self.vertices:
            self[(v1, v2)] = 1
            self[(v2, v1)] = 1

    def add_vertex(self, v):
        self.vertices.append(v)

    def remove_edge(self, v1, v2):
        del self[(v1, v2)]
        del self[(v2, v1)]

    def remove_vertex(self, v):
        keys = list(self.keys())
        for v1, v2 in keys:
            if v1 == v or v2 == v and self.get((v1, v2)):
                del self[(v1, v2)]
        self.vertices.remove(v)


class DictGraphTest(unittest.TestCase):

    def test_add_edge_for_existing_vertices(self):
        graph = DictGraph([1, 2, 3])
        graph.add_edge(1, 2)

        self.assertIsNotNone(graph[(1, 2)])
        self.assertIsNotNone(graph[(2, 1)])

    def test_add_edge_for_not_existing_vertices(self):
        graph = DictGraph([1, 2, 3])
        graph.add_edge(4, 2)

        self.assertIsNone(graph.get((4, 2)))
        self.assertIsNone(graph.get((2, 4)))

    def test_remove_edge(self):
        graph = DictGraph([1, 2, 3])
        graph.add_edge(1, 2)
        graph.add_edge(3, 2)
        graph.remove_edge(3, 2)

        self.assertIsNotNone(graph.get((1, 2)))
        self.assertIsNone(graph.get((3, 2)))
        self.assertIsNone(graph.get((2, 3)))

    def test_remove_vertex(self):
        graph = DictGraph([1, 2, 3])
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)

        graph.remove_vertex(3)
        self.assertNotIn(3, graph.vertices)
        self.assertNotIn((1, 3), graph.keys())
        self.assertIn((1, 2), graph.keys())


def plot_nerwork(G):

    # elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d[nameWeight] > 0.5]
    # esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d[nameWeight] <= 0.5]

    ###positions for all nodes - seed for reproducibility###
    pos = nx.spring_layout(G, seed=1)
    # pos = nx.nx_agraph.graphviz_layout(G)
    # pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    
    
    ###nodes###
    nx.draw_networkx_nodes(G, pos, node_size=70)

    ###edges###
    # nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
    # nx.draw_networkx_edges(
    #    G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    # )
    nameWeights = names[2]
    widths = nx.get_edge_attributes(G, nameWeights)
    nx.draw_networkx_edges(G, pos,
                           edgelist=widths.keys(),
                           width=list(widths.values()),
                           edge_color='lightblue',
                           alpha=0.6)

    ###node labels###
    # nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    # edge weight labels
    # nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def dataframe2network(df, names):
    
    nodeAname = names[0]
    nodeBname = names[1]
    weights = names[2]

    G = nx.from_pandas_edgelist(
        df, nodeAname, nodeBname, edge_attr=weights)
    #durations = [i['STdistance'] for i in dict(G.edges).values()]
    return G

def dataframe2CSV(df, nameFile):
    df.to_csv(nameFile, sep='\t', encoding='utf-8', index = False)



def CSV2dataframe(nameFile, delimiter = '\t'):
    df = pd.read_csv(nameFile, delimiter=delimiter)
    return df


def arrays2dataframe(data, names):
    df = pd.DataFrame(data=data.T,    # values
                 columns=names)  # 1st row as the column names
    return df
    # index=data[1:, 0],    # 1st column as index

def remove_given_edges(network, edges2remove):

    #network_minus = nx.Graph(network)
    #network_minus = network.copy()
    network_minus = network# network.__class__()
    #network_minus.add_nodes_from(network)
    #network_minus.add_edges_from(network.edges)
    network_minus.remove_nodes_from(edges2remove)
    
    return network

def random_knodes_deletion(nodes, proportion2remove):
    '''remove randomly a proportion of nodes
      given a list of nodes and a proportion to remove between 0 (non) and 1 (all)'''
    size = int(len(nodes)*proportion2remove)
   
    nodes2remove = nodes[np.random.choice(len(nodes), size=size, replace=False)]
   
    return nodes2remove


def create_random_net(rangeNodesNumeric, numNodes, weightNorm = 10):
    nodesA = np.random.choice(
        rangeNodesNumeric,  size=(numNodes))  # size=(3, 5, 4)
    nodesB = np.random.choice(rangeNodesNumeric,  size=(numNodes))
    weights = np.random.rand(numNodes) * weightNorm
    data = np.int_([nodesA, nodesB, weights])


rangeNodesNumeric = 22
numNodes = 11
proportion2remove = 0.333  # between 0 (non) and 1 (all)
weightNorm = 10
#create_random_net(rangeNodesNumeric, numNodes, weightNorm)

nameFile = './data/tryNetwork_v01.csv'
#data = [nodesA, nodesB, weights]
names = ['nodeA', 'nodeB', 'weights']

#df = arrays2dataframe(data, names)
#dataframe2CSV(df, nameFile)
df = CSV2dataframe(nameFile)
G = dataframe2network(df, names)


print(G)
print('nodesA', df['nodeA']) 
print('nodesB', df['nodeB'])
nodes = list(G.nodes)
edges = list(G.edges)
print('edges', edges)
print('nodes', nodes)
nodes2remove = random_knodes_deletion(np.array(nodes), proportion2remove)
G_minus = remove_given_edges(G, nodes2remove)
print(G_minus)
print(G)




#plot_nerwork(G)
#plot_nerwork(G_minus)



#dict = {{a:b}:w for a,b,w in zip (nodesA, nodesB, weights)}
#Gclass = DictGraph(dict)

#G = nx.Graph()
#for a, b, w in zip (nodesA, nodesB, weights):
#    G.add_edge(a, b, weight=w)



