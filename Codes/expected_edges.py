import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import itertools
import pickle
from collections import defaultdict
from tqdm import tqdm
'''
Computation of expected edges according to a null model
Datas are saved in the external file expected_edges.pkl
'''
def loopless_unweighted(matrix):
    matrix.data[:] = 1
    matrix.setdiag(0)
    return matrix

def connection_probability(matrix):
    return matrix.sum() / matrix.shape[0]**2

def to_array(my_list):
    my_array = []
    for etype, element in my_list.items():
        my_array.append(element)
    return my_array   
        
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
edge_list_path = 'edges.csv'  # Sostituisci con il percorso effettivo del tuo file
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
type = {}

subgraphs2 = {}

# Adjacency matrix
for subgraph in subgraphs:
    subgraph_copy = subgraph.copy()
    adj_matrix = np.zeros((len(all_nodes), len(all_nodes)))
    for u, v, data in subgraph_copy.edges(data = True):
        type = data['attribute']
        i,j = list(all_nodes).index(u), list(all_nodes).index(v)
        adj_matrix[i, j] = data['weight']
    adj_matrix = sp.csr_matrix(adj_matrix)
    adj_matrix = loopless_unweighted(adj_matrix)
    adj_matrices[type] = adj_matrix
    N = adj_matrix.shape[0]
    subgraphs2[type] = nx.from_scipy_sparse_matrix(adj_matrix)
    
   
    
prob = {}
configurations = {}
i = 0
for etype, matrix in adj_matrices.items():
    x = np.zeros(4)
    x[i] = 1
    configurations[etype] = x
    print(f'Type {etype} connection probability {connection_probability(matrix)}')
    prob[etype] = connection_probability(matrix)
    i += 1
    
expected_links = {}



prob_array = to_array(prob)
with open('probabilities.pkl', 'wb') as file:
    pickle.dump(prob, file)


configurations['all'] = sum(configurations[etype] for etype in adj_matrices)

for combo in itertools.combinations(adj_matrices, 2):
    combo_key = '+'.join(combo)
    configurations[combo_key] = sum(configurations[etype] for etype in combo)
    


for combo in itertools.combinations(adj_matrices, 3):
    combo_key = '+'.join(combo)
    configurations[combo_key] = sum(configurations[etype] for etype in combo)


for etype, config in configurations.items():
    expected_links[etype] = N**2 * combo_prob(config, prob_array)
    
print(expected_links)
with open('expected_edges.pkl', 'wb') as file:
    pickle.dump(expected_links, file)
