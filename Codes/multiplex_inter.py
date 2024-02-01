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


edge_list_path = 'edges.csv'  
df = pd.read_csv(edge_list_path)
nodes_path = 'nodes.csv'
read_nodes = pd.read_csv(nodes_path)

multigraph = nx.MultiDiGraph()
multiplex = nx.DiGraph()
# linking construction
for _, row in df.iterrows():
    source, target, weight, etype = row['source'], row['target'], row['weight'], row['etype']
    multigraph.add_edge(source, target, weight=weight, attribute=etype)



subgraphs = [multigraph.edge_subgraph(((u, v, k) for u, v, k, data in multigraph.edges(keys=True, data=True) if data['attribute'] == etype)) for etype in set(nx.get_edge_attributes(multigraph, 'attribute').values())]
intersection_graph = multigraph.subgraph(set.intersection(*(set(subgraph.nodes) for subgraph in subgraphs)))
subgraph_nodes_sets = [set(subgraph.nodes) for subgraph in subgraphs]

intersection_nodes = set.intersection(*subgraph_nodes_sets)

for u, v, data in intersection_graph.edges(data=True):
    etype = data['attribute']  
    weights = data['weight']
    multiplex.add_edge(u, v, weight = weights, layer=etype)
    
for _, data in read_nodes.iterrows():
    node_id = data['# index']
    hemisphere = data[' hemisphere']
    vids = data[' vids']
    homologue = data[' homologue']
    if multiplex.has_node(node_id):
        multiplex.nodes[node_id]['Hemisphere'] = hemisphere
        multiplex.nodes[node_id]['vids'] = vids
        multiplex.nodes[node_id]['homologue'] = homologue

density = nx.density(multiplex)
degree_sequence = sorted((d for n, d in multiplex.degree()), reverse=True)
max_degree = max(degree_sequence)
nodes_number = nx.number_of_nodes(multiplex)
synapses = sum(weight['weight'] for _, _, weight in multiplex.edges(data=True))
print(f'Total: Number of nodes = {nodes_number} Number of edges = {multiplex.number_of_edges()}  Number of synapses = {synapses} Density = {density} Max Degree = {max_degree}')


nodes = set(multiplex.nodes())        
left_nodes = set(node for node, data in multiplex.nodes(data=True) if data['Hemisphere'] == 'left')
right_nodes = set(node for node, data in multiplex.nodes(data=True) if data['Hemisphere'] == 'right')
print(f'Left nodes= {len(left_nodes)}   Right nodes = {len(right_nodes)}')
all_nodes = len(left_nodes.union(right_nodes))
all_nodes_set = left_nodes.union(right_nodes)
no_hemisphere = nodes-all_nodes_set
cross_hemisphere_nodes = set()
for source, target in multiplex.edges():
    if source in left_nodes and target in right_nodes:
        if source not in cross_hemisphere_nodes:
            cross_hemisphere_nodes.add(source)
        if target not in cross_hemisphere_nodes:
            cross_hemisphere_nodes.add(target)
    elif source in right_nodes and target in left_nodes:
        if source not in cross_hemisphere_nodes:
            cross_hemisphere_nodes.add(source)
        if target not in cross_hemisphere_nodes:
            cross_hemisphere_nodes.add(target)
            
    
involved_nodes = set()

same_hemisphere_connections = 0
for source, target in multiplex.edges():
    if source in left_nodes and target in left_nodes:
        same_hemisphere_connections += 1
        if source not in involved_nodes  :
            involved_nodes.add(source)
        if target not in involved_nodes:
            involved_nodes.add(target)
    elif source in right_nodes and target in right_nodes:
        same_hemisphere_connections += 1
        if source not in involved_nodes  :
            involved_nodes.add(source)
        if target not in involved_nodes:
            involved_nodes.add(target)

contralateral_neurons_set = cross_hemisphere_nodes-cross_hemisphere_nodes.intersection(involved_nodes)
contralateral_neurons = len(contralateral_neurons_set)
bilateral_neurons_set = cross_hemisphere_nodes - contralateral_neurons_set
ipsilateral_nodes_set = all_nodes_set - cross_hemisphere_nodes
bilateral_neurons = len(bilateral_neurons_set)
ipsilateral_neurons = len(ipsilateral_nodes_set)
print('Multiplex')
print(f"Contralateral neurons: {contralateral_neurons}  Fraction = {contralateral_neurons / all_nodes}")
print(f"Bilateral neurons: {bilateral_neurons}  Fraction = {bilateral_neurons / all_nodes}")
print(f"Ipsilateral neurons: {ipsilateral_neurons}  Fraction = {ipsilateral_neurons / all_nodes}")
print(f'Not catalogued: {len(no_hemisphere)}')
print(f'Contralateral neurons:{contralateral_neurons_set}')