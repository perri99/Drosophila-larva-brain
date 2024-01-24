import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp


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
edge_list_path = 'edges.csv'  # Sostituisci con il percorso effettivo del tuo file
df = pd.read_csv(edge_list_path)

multigraph = nx.MultiDiGraph()

# linking construction
for _, row in df.iterrows():
    source, target, weight, etype = row['source'], row['target'], row['weight'], row['etype']
    multigraph.add_edge(source, target, weight=weight, attribute=etype)

# Ottieni l'insieme completo dei nodi del multigrafo
all_nodes = set(multigraph.nodes())
edges = multigraph.edges(data=True)
edge_labels = [data['attribute'] for _, _, data in edges]

subgraphs = [multigraph.edge_subgraph(((u, v, k) for u, v, k, data in multigraph.edges(keys=True, data=True) if data['attribute'] == etype)) for etype in set(nx.get_edge_attributes(multigraph, 'attribute').values())]

adj_matrices = {}
subgraphs2 = {}
edges_set = {}

# Aggiungi eventuali nodi mancanti ai sottografi
for subgraph in subgraphs:
    subgraph_copy = subgraph.copy()
    adj_matrix = np.zeros((len(all_nodes), len(all_nodes)))
    for u, v, data in subgraph_copy.edges(data = True):
        i,j = list(all_nodes).index(u), list(all_nodes).index(v)
        adj_matrix[i, j] = data['weight']
        etype = data['attribute']
    adj_matrix = sp.csr_matrix(adj_matrix)
    plt.spy(adj_matrix, markersize =1 )
    plt.title(etype)
    plt.show()
    adj_matrices[etype] = adj_matrix
    subgraphs2[etype] = nx.from_scipy_sparse_matrix(adj_matrix)
    
