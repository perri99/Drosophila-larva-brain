import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from networkx.algorithms import isomorphism
from scipy.optimize import linear_sum_assignment, quadratic_assignment
from memory_profiler import profile

# reading edges
edge_list_path = 'edges.csv'  
df = pd.read_csv(edge_list_path)

nodes_path = 'nodes.csv'
read_nodes = pd.read_csv(nodes_path)

symmetrical_path = 'edge_symmetry3.csv'
symmetrical_read = pd.read_csv(symmetrical_path)

multigraph = nx.MultiDiGraph()

symmetrical_edges_aa = np.zeros(20)
symmetrical_edges_ad = np.zeros(20)
symmetrical_edges_da = np.zeros(20)
symmetrical_edges_dd = np.zeros(20)
symmetrical_edges_total = np.zeros(20)

for _,row in symmetrical_read.iterrows():
    weight = row['weight-R']
    etype = row['attribute']
    if weight >20:
        weight = 20
    symmetrical_edges_total[weight-1] += 1
    if etype == 'aa':
        symmetrical_edges_aa[weight-1] += 1
    if etype == 'ad':
        symmetrical_edges_ad[weight-1] += 1
    if etype == 'da':
        symmetrical_edges_da[weight-1] += 1
    if etype == 'dd':
        symmetrical_edges_dd[weight-1] += 1
# linking construction
for _, row in df.iterrows():
    source, target, weight, etype = row['source'], row['target'], row['weight'], row['etype']
    multigraph.add_edge(source, target, weight=weight, attribute=etype)

#nodes attribute
for _, data in read_nodes.iterrows():
    node_id = data['# index']
    hemisphere = data[' hemisphere']
    
    # Assicurati che il nodo esista prima di assegnare attributi
    if multigraph.has_node(node_id):
        # Assegna l'attributo 'Hemisphere' al nodo
        multigraph.nodes[node_id]['Hemisphere'] = hemisphere
all_nodes = set(multigraph.nodes())
left_hemisphere = multigraph.subgraph(node for node, data in multigraph.nodes(data=True) if data['Hemisphere'] == 'left')
right_hemisphere = multigraph.subgraph(node for node, data in multigraph.nodes(data=True) if data['Hemisphere'] == 'right')

total_links = right_hemisphere.number_of_edges()
link_counts_aa = np.zeros(20)
link_counts_ad = np.zeros(20)
link_counts_da = np.zeros(20)
link_counts_dd = np.zeros(20)
link_counts_total = np.zeros(20)
for _, _, data in right_hemisphere.edges(data=True):
        etype = data['attribute']
        weight = data['weight']
       
        if weight > 20:
            weight = 20
        link_counts_total[weight-1] += 1
        if etype == 'aa':
            link_counts_aa[weight-1] += 1
        if etype == 'ad':
            link_counts_ad[weight-1] += 1
        if etype == 'da':
            link_counts_da[weight-1] += 1
        if etype == 'dd':
            link_counts_dd[weight-1] += 1
fraction_da = symmetrical_edges_da / link_counts_da
indici_nan = np.isnan(fraction_da)
fraction_da = np.where(indici_nan, 1, fraction_da)
print(symmetrical_edges_aa / link_counts_aa)
print(symmetrical_edges_ad / link_counts_ad)
print(fraction_da)
print(symmetrical_edges_dd / link_counts_dd)
print(symmetrical_edges_total / link_counts_total)
print(link_counts_total)

somma = np.sum(link_counts_total)
symmetrical_sum = np.sum(symmetrical_edges_total)
print(f'Total fraction = {symmetrical_sum/somma}') 

somma_intervallo = np.sum(symmetrical_edges_total[9:19])
somma_intervallo2 = np.sum(link_counts_total[9:19]) 
print(somma_intervallo/somma_intervallo2)

somma_intervallo = np.sum(symmetrical_edges_total[14:19])
somma_intervallo2 = np.sum(link_counts_total[14:19]) 
print(somma_intervallo/somma_intervallo2)

x = np.linspace(1, 20,20)
plt.plot(x, symmetrical_edges_aa / link_counts_aa, label = 'aa')
plt.plot(x, symmetrical_edges_ad / link_counts_ad, label = 'ad')
plt.plot(x, fraction_da, label = 'da')
plt.plot(x, symmetrical_edges_dd / link_counts_dd, label = 'dd')
plt.plot(x, symmetrical_edges_total / link_counts_total, color = 'black', label = 'total')
plt.ylim(0, 1.2)
plt.xlim(1, 20)
plt.xlabel('Weights')
plt.ylabel('Symmetrical Edges fraction')
plt.title("Fraction of symmetrical edges")
plt.legend()
plt.savefig('edge_symmetry.png')
plt.show()

