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
    if multigraph.has_node(node_id):
        multigraph.nodes[node_id]['Hemisphere'] = hemisphere
        multigraph.nodes[node_id]['vids'] = vids
        multigraph.nodes[node_id]['homologue'] = homologue


subgraphs = [multigraph.edge_subgraph(((u, v, k) for u, v, k, data in multigraph.edges(keys=True, data=True) if data['attribute'] == etype)) for etype in set(nx.get_edge_attributes(multigraph, 'attribute').values())]

left_nodes = set(node for node, data in multigraph.nodes(data=True) if data['Hemisphere'] == 'left')
right_nodes = set(node for node, data in multigraph.nodes(data=True) if data['Hemisphere'] == 'right')
all_nodes = len(left_nodes.union(right_nodes))
all_nodes_set = left_nodes.union(right_nodes)

# Identifica i nodi coinvolti nei collegamenti tra emisferi diversi
cross_hemisphere_nodes = set()
for source, target in multigraph.edges():
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
            
    # Trova i nodi collegati solamente con nodi dell'altro emisfero
    
# Inizializza un insieme per tenere traccia dei nodi coinvolti nei collegamenti
involved_nodes = set()

# Conta il numero di nodi coinvolti nei collegamenti nello stesso emisfero
same_hemisphere_connections = 0
for source, target in multigraph.edges():
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

    # Calcola il numero di nodi coinvolti
contralateral_neurons_set = cross_hemisphere_nodes-cross_hemisphere_nodes.intersection(involved_nodes)
contralateral_neurons = len(contralateral_neurons_set)
bilateral_neurons_set = cross_hemisphere_nodes - contralateral_neurons_set
ipsilateral_nodes_set = all_nodes_set - cross_hemisphere_nodes
bilateral_neurons = len(bilateral_neurons_set)
ipsilateral_neurons = len(ipsilateral_nodes_set)
print('Entire connectome')
print(f"Contralateral neurons: {contralateral_neurons}  Fraction = {contralateral_neurons / all_nodes}")
print(f"Bilateral neurons: {bilateral_neurons}  Fraction = {bilateral_neurons / all_nodes}")
print(f"Ipsilateral neurons: {ipsilateral_neurons}  Fraction = {ipsilateral_neurons / all_nodes}")

for graph in subgraphs:
    left_nodes = set(node for node, data in graph.nodes(data=True) if data['Hemisphere'] == 'left')
    right_nodes = set(node for node, data in graph.nodes(data=True) if data['Hemisphere'] == 'right')
    all_nodes = len(left_nodes.union(right_nodes))
    all_nodes_set = left_nodes.union(right_nodes)
    for u, v, data in graph.edges(data = True):
        etype = data['attribute']
    # Identifica i nodi coinvolti nei collegamenti tra emisferi diversi
    cross_hemisphere_nodes = set()
    for source, target in graph.edges():
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
            
    # Trova i nodi collegati solamente con nodi dell'altro emisfero
    exclusive_cross_hemisphere_nodes = set()
    for node in cross_hemisphere_nodes:
        neighbors = set(graph.neighbors(node))
        if all(n in (left_nodes | right_nodes) - {node} for n in neighbors) and node not in exclusive_cross_hemisphere_nodes:
            exclusive_cross_hemisphere_nodes.add(node)

    # Inizializza un insieme per tenere traccia dei nodi coinvolti nei collegamenti
    involved_nodes = set()

    # Conta il numero di nodi coinvolti nei collegamenti nello stesso emisfero
    same_hemisphere_connections = 0
    for source, target in graph.edges():
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

    # Calcola il numero di nodi coinvolti
    contralateral_neurons_set = cross_hemisphere_nodes-cross_hemisphere_nodes.intersection(involved_nodes)
    contralateral_neurons = len(contralateral_neurons_set)
    bilateral_neurons_set = cross_hemisphere_nodes - contralateral_neurons_set
    ipsilateral_nodes_set = all_nodes_set - cross_hemisphere_nodes
    bilateral_neurons = len(bilateral_neurons_set)
    ipsilateral_neurons = len(ipsilateral_nodes_set)
    # Stampa il risultato
    print(etype)
    print(all_nodes)
    print(f"Contralateral neurons: {contralateral_neurons}  Fraction = {contralateral_neurons / all_nodes}")
    print(f"Bilateral neurons: {bilateral_neurons}  Fraction = {bilateral_neurons / all_nodes}")
    print(f"Ipsilateral neurons: {ipsilateral_neurons}  Fraction = {ipsilateral_neurons / all_nodes}")

    print((contralateral_neurons + bilateral_neurons + ipsilateral_neurons) / all_nodes) 
# Identifica i nodi di ciascun emisfero
