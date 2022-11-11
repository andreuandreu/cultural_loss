import networkx as nx
import pandas as pd
import dash
import matplotlib.pyplot as plt

from smallworld.draw import draw_network
from smallworld import get_smallworld_graph


# define network parameters
N = 21
k_over_2 = 2
betas = [0, 0.025, 1.0]
labels = [r'$\beta=0$', r'$\beta=0.025$', r'$\beta=1$']

focal_node = 0


fig, ax = plt.subplots(1, 3, figsize=(9, 3))

def small_world():

    # scan beta values
    for ib, beta in enumerate(betas):

        # generate small-world graphs and draw
        G = get_smallworld_graph(N, k_over_2, beta)
        draw_network(G, k_over_2, focal_node=focal_node, ax=ax[ib])

        ax[ib].set_title(labels[ib], fontsize=11)

    # show

graph = nx.Graph()

print (graph.nodes())
print (graph.edges())


graph.add_nodes_from([
    (0, {"color": "green"}),
    (1, {"color": "red"}),
    (2, {"color": "red"}),
    (3, {"color": "red"}),
    (4, {"color": "red"}),
    (5, {"color": "red"}),
])

graph.add_edges_from(
    [(0, 1), (2,3), (4,5)], color = "green"
)

#path graph
graphH = nx.path_graph(3)

graph.add_nodes_from(graphH)
graph.add_edges_from(graphH.edges)

# petersen
GP = nx.petersen_graph()

graph.add_nodes_from(GP)
graph.add_edges_from(GP.edges)

#small world
SM = get_smallworld_graph(N, k_over_2, 0.05)
graph.add_nodes_from(SM)
graph.add_edges_from(SM.edges)



#graphH.clear()

graph[1][2]['color'] = "blue"

print(graph.nodes())
print(graph.edges())

print(graph[1][2])
print(graph.edges[1, 2])

#graph.remove_node(2)
#graph.remove_nodes_from(graphH)

positions = nx.circular_layout(graph)


nx.draw(graph, positions,  with_labels=True, ax = ax[0])


draw_network(graph, k_over_2, focal_node=focal_node,
             ax=ax[1])  # font_weight='bold'

# another inception of small world, k nearest neighbours, n nodes number, p rewiring prob


G = nx.Graph()#nx.watts_strogatz_graph(n=10, k=4, p=0.1)

# add weighted edges
E = [('A', 'B', 2), ('A', 'C', 1), ('B', 'D', 5), ('B', 'E', 3), ('C', 'E', 2)]
G.add_weighted_edges_from(E)

#pos = nx.spring_layout(G)
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold', ax = ax[2])
edge_weight = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weight, ax=ax[2])

plt.show()

# save the graph to plot with other external tools
path = './data/output/toy_graph'
nx.write_graphml(G, path)
