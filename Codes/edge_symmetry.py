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

def graph_matching(graph1, graph2):
    # Calcola la distanza di modifica tra i due grafi
    adj_matrix1 = nx.to_numpy_array(graph1)
    adj_matrix2 = nx.to_numpy_array(graph2)
    
    min_dim = min(adj_matrix1.shape[0], adj_matrix2.shape[0])
    
    # Ridimensiona entrambe le matrici alla dimensione minima
    
    adj_matrix1 = adj_matrix1[:min_dim, :min_dim]
    adj_matrix2 = adj_matrix2[:min_dim, :min_dim]
     
    num_random_inits = 1
    '''
    for u, v, data in graph1.edges(data = True):
        i,j = list(all_nodes).index(u), list(all_nodes).index(v)
        adj_matrix1[i, j] = data['weight']
    
    for u, v, data in graph2.edges(data = True):
        i,j = list(all_nodes).index(u), list(all_nodes).index(v)
        adj_matrix2[i, j] = data['weight']
    '''
    left_nodes = list(graph1.nodes())
    right_nodes = list(graph2.nodes())
    #common_nodes = left_nodes & right_nodes
    
    
    for init in tqdm(range(num_random_inits)):
        #options = {"partial_match": seed_pairs}
        result = quadratic_assignment(adj_matrix1, adj_matrix2, method='faq')

        # Ottieni gli indici delle corrispondenze tra i nodi
        node_matching = result['col_ind']
        objective_value = result['fun']
        print(f'Objective value: {objective_value}')
        #predicted_pairs = [(left_nodes[i], right_nodes[matching_indices[i]]) for i in range(len(left_nodes))]
        #seed_pairs.extend(predicted_pairs)
    # Stampa il risultato
    np.savetxt("L_R.txt", node_matching)
    
    print("Corrispondenze tra i nodi:")
    for idx, match in enumerate(node_matching):
        print(f"Nodo {idx} nel grafo A corrisponde a Nodo {match} nel grafo B")

def mutual_reciprocity(source,target):
    source.data[:] = 1
    target.data[:] = 1   #unweighted
    source.setdiag(0)    #loopless
    target.setdiag(0)
    sum = source.sum()
    prod = (source.multiply(target.T)).sum()
    return prod/sum

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
# Creazione dei quattro sottografi
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