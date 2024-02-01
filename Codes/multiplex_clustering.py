import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import community  
import pickle
# reading edges
edge_list_path = 'edges.csv'  
df = pd.read_csv(edge_list_path)
nodes_path = 'nodes.csv'
read_nodes = pd.read_csv(nodes_path)

multigraph = nx.MultiDiGraph()
multiplex = nx.DiGraph()
# linking construction
for _, row in df.iterrows():
    source, target, weight, etype = row['source'], row['target'], row['weight'], row['etype']
    multigraph.add_edge(source, target, weight=weight, attribute=etype)


subgraphs = [multigraph.edge_subgraph(((u, v, k) for u, v, k, data in multigraph.edges(keys=True, data=True) if data['attribute'] == etype)) for etype in set(nx.get_edge_attributes(multigraph, 'attribute').values())]
intersection_graph = multigraph.subgraph(set.intersection(*(set(subgraph.nodes) for subgraph in subgraphs)))

for u, v, data in intersection_graph.edges(data=True):
    etype = data['attribute']  
    weights = data['weight']
    multiplex.add_edge(u, v, weight = weights, layer=etype)

#nodes attribute
for _, data in read_nodes.iterrows():
    node_id = data['# index']
    hemisphere = data[' hemisphere']
    vids = data[' vids']
    homologue = data[' homologue']
    if multiplex.has_node(node_id):
        multiplex.nodes[node_id]['Hemisphere'] = hemisphere
        multiplex.nodes[node_id]['vids'] = vids
        multiplex.nodes[node_id]['homologue'] = homologue


communities = nx.community.greedy_modularity_communities(multiplex)
supergraph = nx.cycle_graph(len(communities))
superpos = nx.spring_layout(multiplex, scale=50, seed=429)


colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


centers = list(superpos.values())
pos = {}
print(communities)
for center, comm in zip(centers, communities):
    pos.update(nx.spring_layout(nx.subgraph(multiplex, comm), center=center, seed=1430))

for i, nodes in enumerate(communities):
    nx.draw_networkx_nodes(multiplex, pos=pos, nodelist=nodes, node_color=colors[i], node_size=100)

nx.draw_networkx_edges(multiplex, alpha=0.05, pos=pos)

plt.tight_layout()
plt.savefig('multiplex_clustering.png')
plt.show()

with open('communities.pkl', 'wb') as file:
    pickle.dump(communities, file)