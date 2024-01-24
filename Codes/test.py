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

# Funzione per calcolare la probabilità di connessione
def calculate_connection_probability(category1, category2, graph):
    total_connections = 0
    total_possible_connections = 0

    # Itera su tutte le coppie di neuroni tra le due categorie
    for node1 in category1:
        for node2 in category2:
            # Controlla se c'è una connessione tra i neuroni
            if graph.has_edge(node1, node2):
                total_connections += 1

            # Aumenta il numero totale di possibili connessioni
            total_possible_connections += 1

    # Calcola la probabilità di connessione
    connection_probability = total_connections / total_possible_connections

    return connection_probability

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
    
    bilateral_neurons_set = cross_hemisphere_nodes - contralateral_neurons_set
    ipsilateral_nodes_set = all_nodes_set - cross_hemisphere_nodes
    
    
    
    print(f'Connection: {etype}')
    ipsi_ipsi_probability = calculate_connection_probability(ipsilateral_nodes_set, ipsilateral_nodes_set, graph)
    print(f"Probabilità di connessione tra neuroni ipsi-ipsi: {ipsi_ipsi_probability}")
    ipsi_bi_probability = calculate_connection_probability(ipsilateral_nodes_set, bilateral_neurons_set, graph)
    print(f"Probabilità di connessione tra neuroni ipsi-bi: {ipsi_bi_probability}")
    ipsi_contra_probability = calculate_connection_probability(ipsilateral_nodes_set, contralateral_neurons_set, graph)
    print(f"Probabilità di connessione tra neuroni ipsi-contra: {ipsi_contra_probability}")
    contra_contra_probability = calculate_connection_probability(contralateral_neurons_set, contralateral_neurons_set, graph)
    print(f"Probabilità di connessione tra neuroni contra-contra: {contra_contra_probability}")
    bi_contra_probability = calculate_connection_probability(bilateral_neurons_set, contralateral_neurons_set, graph)
    print(f"Probabilità di connessione tra neuroni bi-contra: {bi_contra_probability}")
    bi_bi_probability = calculate_connection_probability(bilateral_neurons_set, bilateral_neurons_set, graph)
    print(f"Probabilità di connessione tra neuroni bi-bi: {bi_bi_probability}")
    
    homologous_contralateral_pairs = set()
    contralateral_pairs = set()
    homolog_count = 0
    
    # Identifica coppie di neuroni contralaterali omologhi sinistra-destra
    for u in left_nodes:
        if u in contralateral_neurons_set:
            u_homo = graph.nodes[u]['homologue']
            for v in right_nodes:
                v_vids = graph.nodes[v]['vids']
                if v_vids == u_homo and (u,v) not in homologous_contralateral_pairs:
                    homologous_contralateral_pairs.add((u,v))
    for u in right_nodes:
        if u in contralateral_neurons_set:
            u_homo = graph.nodes[u]['homologue']
            for v in left_nodes:
                v_vids = graph.nodes[v]['vids']
                if v_vids == u_homo and (u,v) not in homologous_contralateral_pairs:
                    homologous_contralateral_pairs.add((u,v)) 
                    
    for pair in homologous_contralateral_pairs:
        left, right = pair
        if graph.has_edge(left, right):
            homolog_count +=1
    contralateral_edges_homo = len(homologous_contralateral_pairs)
    
    
    # Calcola la probabilità di sinaptizzazione diretta tra neuroni omologhi
    total_homologous_pairs = contralateral_edges_homo
    direct_synapses_probability = homolog_count / contralateral_edges_homo

    print(f"Coppie omologhe con neurone controlaterale: {total_homologous_pairs}")
    print(f"Numero di coppie omologhe contralaterali con sinaptizzazione diretta: {homolog_count}")
    print(f"Probabilità di sinaptizzazione diretta tra neuroni omologhi contralaterali: {direct_synapses_probability}")
   