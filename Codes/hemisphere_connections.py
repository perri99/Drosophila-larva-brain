import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from networkx.algorithms import isomorphism
from scipy.sparse import csr_matrix
from tqdm import tqdm
from matplotlib.lines import Line2D
import csv
'''
Hemispheres subgraphs
'''

# reading edges
edge_list_path = 'edges.csv'  
df = pd.read_csv(edge_list_path)
nodes_path = 'nodes.csv'
read_nodes = pd.read_csv(nodes_path)

multigraph = nx.MultiDiGraph()

# linking construction
for _, row in df.iterrows():
    source, target, weight, etype = row['source'], row['target'], row['weight'], row['etype']
    multigraph.add_edge(source, target, weight=weight, attribute=etype)

#nodes attribute
for _, data in read_nodes.iterrows():
    node_id = data['# index']
    hemisphere = data[' hemisphere']
    vids = data[' vids']
    homologue = data[' homologue']
    # Assicurati che il nodo esista prima di assegnare attributi
    if multigraph.has_node(node_id):
        # Assegna l'attributo 'Hemisphere' al nodo
        multigraph.nodes[node_id]['Hemisphere'] = hemisphere
        multigraph.nodes[node_id]['vids'] = vids
        multigraph.nodes[node_id]['homologue'] = homologue

all_nodes = set(multigraph.nodes())
all_nodes = sorted(all_nodes, key=lambda node: (str(multigraph.nodes[node]['Hemisphere']), node))
#subgraphs
left_left_subgraph = nx.MultiDiGraph()
right_right_subgraph = nx.MultiDiGraph()
left_right_subgraph = nx.MultiDiGraph()
right_left_subgraph = nx.MultiDiGraph()

left_left   = np.zeros((len(all_nodes), len(all_nodes)))
left_right  = np.zeros((len(all_nodes), len(all_nodes)))
right_left  = np.zeros((len(all_nodes), len(all_nodes)))
right_right = np.zeros((len(all_nodes), len(all_nodes)))

for u, v, key, data in multigraph.edges(data=True, keys=True):
    u_hemisphere = multigraph.nodes[u]['Hemisphere']
    v_hemisphere = multigraph.nodes[v]['Hemisphere']

    if u_hemisphere == 'left' and v_hemisphere == 'left':
        left_left_subgraph.add_edge(u, v, key=key, weight=data['weight'], attribute=data['attribute'])
        i,j = list(all_nodes).index(u), list(all_nodes).index(v)
        left_left[i,j] = data['weight']
    elif u_hemisphere == 'right' and v_hemisphere == 'right':
        right_right_subgraph.add_edge(u, v, key=key, weight=data['weight'], attribute=data['attribute'])
        a,b = list(all_nodes).index(u), list(all_nodes).index(v)
        right_right[a,b] = data['weight']
    elif u_hemisphere == 'left' and v_hemisphere == 'right':
        left_right_subgraph.add_edge(u, v, key=key, weight=data['weight'], attribute=data['attribute'])
        c,d = list(all_nodes).index(u), list(all_nodes).index(v)
        left_right[c,d] = data['weight']
    elif u_hemisphere == 'right' and v_hemisphere == 'left':
        right_left_subgraph.add_edge(u, v, key=key, weight=data['weight'], attribute=data['attribute'])
        k,l = list(all_nodes).index(u), list(all_nodes).index(v)
        right_left[k,l] = data['weight']


left_left   = csr_matrix(left_left)
left_right  = csr_matrix(left_right)
right_left  = csr_matrix(right_left)
right_right = csr_matrix(right_right)



left_synapses = sum(weight['weight'] for _, _, weight in left_left_subgraph.edges(data=True))
right_synapses = sum(weight['weight'] for _, _, weight in right_right_subgraph.edges(data=True))
left_to_right_synapses = sum(weight['weight'] for _, _, weight in left_right_subgraph.edges(data=True))
right_to_left_synapses = sum(weight['weight'] for _, _, weight in right_left_subgraph.edges(data=True))

print(f'Right-Right: Total number of nodes = {right_right_subgraph.number_of_nodes()}\n Total number of edges = {right_right_subgraph.number_of_edges()}\n Total number of synapses = {right_synapses}')
print(f'Left-Left: Total number of nodes = {left_left_subgraph.number_of_nodes()}\n Total number of edges = {left_left_subgraph.number_of_edges()}\n Total number of synapses = {left_synapses}')
print(f'Right-Left: Total number of nodes = {right_left_subgraph.number_of_nodes()}\n Total number of edges = {right_left_subgraph.number_of_edges()}\n Total number of synapses = {right_to_left_synapses}')
print(f'Left-Right: Total number of nodes = {left_right_subgraph.number_of_nodes()}\n Total number of edges = {left_right_subgraph.number_of_edges()}\n Total number of synapses = {left_to_right_synapses}')

plt.title("Connection between the two hemispheres")
plt.spy(left_left, color = 'blue', markersize=0.05)
plt.spy(left_right, color = 'red' , markersize=0.05)
plt.spy(right_left,  color = 'orange', markersize=0.05)
plt.spy(right_right,  color = 'black', markersize=0.05)
legend_elements = [
    Line2D([0], [0], color='blue', marker='o', markersize=5, label='L-L'),
    Line2D([0], [0], color='red', marker='o', markersize=5, label='L-R'),
    Line2D([0], [0], color='orange', marker='o', markersize=5, label='R-L'),
    Line2D([0], [0], color='black', marker='o', markersize=5, label='R-R')
]

# Aggiunta della legenda utilizzando le linee personalizzate
plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.savefig('combo_hemisphere.png')
