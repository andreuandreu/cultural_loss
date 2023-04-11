

#The Graph class has a constructor that takes in a single argument num_nodes,
# which specifies the number of nodes in the graph. 
# The add_edge method adds an edge between two nodes, with a specified weight. 
# The get_neighbors method returns a dictionary of the neighbors of a given node, 
# with the keys being the neighbor nodes and the values being the weights of the edges
# between the node and its neighbors.





class Graph:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.edges = {}

    def add_edge(self, node1, node2, weight):
        if node1 not in self.edges:
            self.edges[node1] = {}
        if node2 not in self.edges:
            self.edges[node2] = {}
        self.edges[node1][node2] = weight
        self.edges[node2][node1] = weight

    def get_neighbors(self, node):
        return self.edges[node]

# Example usage
#graph = Graph(4)
#graph.add_edge(0, 1, 10)
#graph.add_edge(0, 2, 20)
#graph.add_edge(0, 3, 30)
#graph.add_edge(1, 2, 40)
#print(graph.get_neighbors(0))  # Output: {1: 10, 2: 20, 3: 30}
#print(graph.get_neighbors(1))  # Output: {0: 10, 2: 40}



import matplotlib.pyplot as plt
import networkx as nx

# Create a graph using the Graph class defined above
graph = Graph(4)
graph.add_edge(0, 1, 10)
graph.add_edge(0, 2, 20)
graph.add_edge(0, 3, 30)
graph.add_edge(1, 2, 40)

# Create a NetworkX graph object
G = nx.Graph()

# Add the nodes to the graph
for i in range(graph.num_nodes):
    G.add_node(i)

# Add the edges to the graph
for node, neighbors in graph.edges.items():
    for neighbor, weight in neighbors.items():
        G.add_edge(node, neighbor, weight=weight)

# Use NetworkX to draw the graph
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True)

# Display the plot
plt.show()
