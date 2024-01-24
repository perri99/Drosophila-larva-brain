import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
'''
Graph metrics for the four etype subgraphs and mutual reciprocity 
'''

def mutual_reciprocity(source,target):
    source.data[:] = 1
    target.data[:] = 1   #unweighted
    source.setdiag(0)    #loopless
    target.setdiag(0)
    sum = source.sum()
    prod = (source.multiply(target.T)).sum()
    return prod/sum

def synapses_fraction(matrix):
    matrix.setdiag(0)
    total_synapse = matrix.getnnz()
    forward_matrix = sp.triu(matrix)
    forward_synapses = forward_matrix.getnnz()
    #backward_matrix = sp.tril(matrix)
    backward_synapses = total_synapse-forward_synapses
    return forward_synapses/total_synapse, backward_synapses/total_synapse
    
# reading edges
edge_list_path = 'edges.csv' 
df = pd.read_csv(edge_list_path)

multigraph = nx.MultiDiGraph()

# linking construction
for _, row in df.iterrows():
    source, target, weight, etype = row['source'], row['target'], row['weight'], row['etype']
    multigraph.add_edge(source, target, weight=weight, attribute=etype)


all_nodes = set(multigraph.nodes())
total_synapses = sum(weight['weight'] for _, _, weight in multigraph.edges(data=True))

subgraphs = [multigraph.edge_subgraph(((u, v, k) for u, v, k, data in multigraph.edges(keys=True, data=True) if data['attribute'] == etype)) for etype in set(nx.get_edge_attributes(multigraph, 'attribute').values())]
print(f'Total number of nodes = {multigraph.number_of_nodes()}\n Total number of edges = {multigraph.number_of_edges()}\n Total number of synapses = {total_synapses}')
#number of nodes, density, max degree
for subgraph in subgraphs:

    density = nx.density(subgraph)
    degree_sequence = sorted((d for n, d in subgraph.degree()), reverse=True)
    max_degree = max(degree_sequence)
    nodes_number = nx.number_of_nodes(subgraph)
    etype = list(nx.get_edge_attributes(subgraph, 'attribute').values())
    synapses = sum(weight['weight'] for _, _, weight in subgraph.edges(data=True))
    print(f'Link type: {etype[0]} Number of nodes = {nodes_number} Number of edges = {subgraph.number_of_edges()}  Number of synapses = {synapses} Density = {density*100} Max Degree = {max_degree}')

adj_matrices = []

for subgraph in subgraphs:
    subgraph_copy = subgraph.copy()
    adj_matrix = np.zeros((len(all_nodes), len(all_nodes)))
    for u, v, data in subgraph_copy.edges(data = True):
        i,j = list(all_nodes).index(u), list(all_nodes).index(v)
        adj_matrix[i, j] = data['weight']
    adj_matrix = sp.csr_matrix(adj_matrix)
    adj_matrices.append(adj_matrix)

# Reciprocities
for etype_source, matrix_source in zip(set(nx.get_edge_attributes(multigraph, 'attribute').values()), adj_matrices):
  
    for etype_target, matrix_target in zip(set(nx.get_edge_attributes(multigraph, 'attribute').values()), adj_matrices):
        #matrix_target = matrix_target.toarray()
        mut_rec = mutual_reciprocity(matrix_source, matrix_target)
        print("Source "+ etype_source + " Target " + etype_target)
        print(f"Mutual reciprocity: {mut_rec}")
