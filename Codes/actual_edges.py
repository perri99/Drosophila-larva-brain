import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import itertools
import pickle
from collections import defaultdict
'''
This programs compute the number of the actual edges in the connectome by combination of etypes
and save the output as a dictionary on an external file 'actaul_edges.pkl'
'''
def to_array(my_list):
    my_array = []
    for etype, element in my_list.items():
        my_array.append(element)
    return my_array 

def connection_probability(matrix):
    return matrix.sum() / matrix.shape[0]**2    

def combo_prob(config_array, prob_array):
    product = 1
    for i in range(config_array.shape[0]):
        product *= prob_array[i] ** config_array[i] * (1-prob_array[i]) ** (1-config_array[i])
    return product    

def counting_links(config_array, edges_array):
    common_edges = set()
    
    for i in range(len(config_array)):
        if config_array[i] == 1:
            common_edges = edges_array[i]
            break
            
    for j in range(len(config_array)): 
        if config_array[j] == 1:
            common_edges = common_edges.intersection(edges_array[j])
    
    common_number = len(common_edges)
    return common_number
# reading edges
edge_list_path = 'edges.csv'  
df = pd.read_csv(edge_list_path)

multigraph = nx.MultiDiGraph()

# linking construction
for _, row in df.iterrows():
    source, target, weight, etype = row['source'], row['target'], row['weight'], row['etype']
    multigraph.add_edge(source, target, weight=weight, attribute=etype)

all_nodes = set(multigraph.nodes())
edges = multigraph.edges(data=True)
edge_labels = [data['attribute'] for _, _, data in edges]

subgraphs = [multigraph.edge_subgraph(((u, v, k) for u, v, k, data in multigraph.edges(keys=True, data=True) if data['attribute'] == etype)) for etype in set(nx.get_edge_attributes(multigraph, 'attribute').values())]

adj_matrices = {}
subgraphs2 = {}
edges_set = {}
for subgraph in subgraphs:
    subgraph_copy = subgraph.copy()
    adj_matrix = np.zeros((len(all_nodes), len(all_nodes)))
    for u, v, data in subgraph_copy.edges(data = True):
        i,j = list(all_nodes).index(u), list(all_nodes).index(v)
        adj_matrix[i, j] = data['weight']
        etype = data['attribute']
    adj_matrix = sp.csr_matrix(adj_matrix)
  
    adj_matrices[etype] = adj_matrix
    subgraphs2[etype] = nx.from_scipy_sparse_matrix(adj_matrix)
    
    edges_set[etype] = set(subgraphs2[etype].edges())

configurations = {}
i = 0
for etype, matrix in adj_matrices.items():
    x = np.zeros(4)
    x[i] = 1
    configurations[etype] = x
    i += 1

# sum all the combinations of etype ('aa', 'ad', 'da', 'dd')
configurations['all'] = sum(configurations[etype] for etype in adj_matrices)

for combo in itertools.combinations(adj_matrices, 2):
    combo_key = '+'.join(combo)
    configurations[combo_key] = sum(configurations[etype] for etype in combo)
    

edges_array = to_array(edges_set)
# triplets
for combo in itertools.combinations(adj_matrices, 3):
    combo_key = '+'.join(combo)
    configurations[combo_key] = sum(configurations[etype] for etype in combo)
link_counts = {}

for etype, config in configurations.items():
    link_counts[etype] = counting_links(config, edges_array)
    


print(link_counts)
with open('actual_edges.pkl', 'wb') as file:
    pickle.dump(link_counts, file)

