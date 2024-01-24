import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from networkx.algorithms import isomorphism
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment, quadratic_assignment
from tqdm import tqdm
import numba as nb
import csv

symmetrical_file = 'edge_symmetry3.csv'

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

      
left_left_subgraph = nx.MultiDiGraph()
right_right_subgraph = nx.MultiDiGraph()

for u, v, key, data in multigraph.edges(data=True, keys=True):
    u_hemisphere = multigraph.nodes[u]['Hemisphere']
    v_hemisphere = multigraph.nodes[v]['Hemisphere']
    u_vids = multigraph.nodes[u]['vids']
    v_vids = multigraph.nodes[v]['vids']
    u_h = multigraph.nodes[u]['homologue']
    v_h = multigraph.nodes[v]['homologue']
    if u_hemisphere == 'left' and v_hemisphere == 'left':
        left_left_subgraph.add_edge(u, v, key=key, weight=data['weight'], attribute=data['attribute'])
        left_left_subgraph.nodes[u]['vids'] = u_vids
        left_left_subgraph.nodes[u]['homologue'] = u_h
        left_left_subgraph.nodes[v]['vids'] = v_vids
        left_left_subgraph.nodes[v]['homologue'] = v_h
    elif u_hemisphere == 'right' and v_hemisphere == 'right':
        right_right_subgraph.add_edge(u, v, key=key, weight=data['weight'], attribute=data['attribute'])
        right_right_subgraph.nodes[u]['vids'] = u_vids
        right_right_subgraph.nodes[u]['homologue'] = u_h
        right_right_subgraph.nodes[v]['vids'] = v_vids
        right_right_subgraph.nodes[v]['homologue'] = v_h

# Apri il file CSV in modalità scrittura
with open(symmetrical_file, 'w', newline='') as csvfile:
    # Definisci i nomi delle colonne nel file CSV
    fieldnames = [ 'weight-L','weight-R', 'attribute']

    # Crea il writer CSV
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Scrivi l'intestazione
    writer.writeheader()
    
    for u, v, key, data in tqdm(right_right_subgraph.edges(data=True, keys=True)):
        etype = data['attribute']
        weight_R = data['weight']
        u_homologus = right_right_subgraph.nodes[u]['homologue']
        v_homologus = right_right_subgraph.nodes[v]['homologue']
        for i, j, key2, data2 in left_left_subgraph.edges(data = True, keys = True):
            i_vids = left_left_subgraph.nodes[i]['vids']
            j_vids = left_left_subgraph.nodes[j]['vids']
            if i_vids == u_homologus and j_vids == v_homologus and data2['attribute'] == etype:
                weight_L = data2['weight']
                writer.writerow({ 'weight-L': weight_L, 'weight-R': weight_R, 'attribute': etype})
                

