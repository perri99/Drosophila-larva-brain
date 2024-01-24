import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

def loopless_unweighted(matrix):
    matrix.data[:] = 1
    matrix.setdiag(0)
    return matrix

#reading edges
edge_list_path = 'edges.csv'  # Sostituisci con il percorso effettivo del tuo file
df = pd.read_csv(edge_list_path)

multigraph = nx.MultiDiGraph()

# linking construction
for _, row in df.iterrows():
    source, target, weight, etype = row['source'], row['target'], row['weight'], row['etype']
    multigraph.add_edge(source, target, weight=weight, attribute=etype)

# Ottieni l'insieme completo dei nodi del multigrafo
all_nodes = set(multigraph.nodes())
subgraphs = [multigraph.edge_subgraph(((u, v, k) for u, v, k, data in multigraph.edges(keys=True, data=True) if data['attribute'] == etype)) for etype in set(nx.get_edge_attributes(multigraph, 'attribute').values())]

adj_matrices = {}
# Aggiungi eventuali nodi mancanti ai sottografi
# Aggiungi eventuali nodi mancanti ai sottografi
for subgraph in subgraphs:
    subgraph_copy = subgraph.copy()
    adj_matrix = sp.lil_matrix((len(all_nodes), len(all_nodes)))
    for u, v, data in subgraph_copy.edges(data = True):
        i,j = list(all_nodes).index(u), list(all_nodes).index(v)
        adj_matrix[i, j] = data['weight']
        etype = data['attribute']
    adj_matrix = adj_matrix.tocsr()
    adj_matrices[etype] = adj_matrix
    
for etype_source, matrix_source in adj_matrices.items():
    plt.spy(matrix_source, markersize=0.1, aspect='equal')
    plt.title(f'Adjacency matrix {etype_source} graph')
    plt.savefig(f'Adjacency matrix {etype_source}.png')
    plt.clf()
    
    
 
