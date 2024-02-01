import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
import community  

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
subgraph_nodes_sets = [set(subgraph.nodes) for subgraph in subgraphs]


intersection_nodes = set.intersection(*subgraph_nodes_sets)

for u, v, data in intersection_graph.edges(data=True):
    etype = data['attribute']  
    weights = data['weight']
    multiplex.add_edge(u, v, weight = weights, layer=etype)

for _, data in read_nodes.iterrows():
    node_id = data['# index']
    hemisphere = data[' hemisphere']
    vids = data[' vids']
    homologue = data[' homologue']
    if multiplex.has_node(node_id):
        multiplex.nodes[node_id]['Hemisphere'] = hemisphere
        multiplex.nodes[node_id]['vids'] = vids
        multiplex.nodes[node_id]['homologue'] = homologue
        

density = nx.density(multiplex)
degree_sequence = sorted((d for n, d in multiplex.degree()), reverse=True)
max_degree = max(degree_sequence)
nodes_number = nx.number_of_nodes(multiplex)
synapses = sum(weight['weight'] for _, _, weight in multiplex.edges(data=True))
print(f' Number of nodes = {nodes_number} Number of edges = {multiplex.number_of_edges()}  Number of synapses = {synapses} Density = {density} Max Degree = {max_degree}')



degree_centrality = nx.degree_centrality(multiplex)
closeness_centrality = nx.closeness_centrality(multiplex)
betweenness_centrality = nx.betweenness_centrality(multiplex)
eigenvector_centrality = nx.eigenvector_centrality(multiplex)
pagerank = nx.pagerank(multiplex)
clustering_coefficient = nx.clustering(multiplex)

# Degree Distribution
degree_distribution = dict(multiplex.degree())


node_data = pd.DataFrame({
    'Node': list(multiplex.nodes()),
    'Degree_Centrality': list(degree_centrality.values()),
    'Closeness_Centrality': list(closeness_centrality.values()),
    'Betweenness_Centrality': list(betweenness_centrality.values()),
    'Eigenvector_Centrality': list(eigenvector_centrality.values()),
    'PageRank': list(pagerank.values()),
    
    
    
})


features = ['Degree_Centrality', 'Closeness_Centrality', 'Betweenness_Centrality',
            'Eigenvector_Centrality', 'PageRank']
X = node_data[features]


pca = PCA()
principal_components = pca.fit_transform(X)

node_data['PC1'] = principal_components[:, 0]
node_data['PC2'] = principal_components[:, 1]
print(pca.explained_variance_ratio_)


hemisphere_color_map = {'left': 'blue', 'right': 'red', 'nan': 'green'}

filtered_hemisphere_colors = []
default_color = 'gray'  

for node in multiplex.nodes:
    hemisphere = multiplex.nodes[node].get('Hemisphere', 'nan')
    color = hemisphere_color_map.get(hemisphere, default_color)
    filtered_hemisphere_colors.append(color)

plt.scatter(node_data['Degree_Centrality'], node_data['Closeness_Centrality'], c=filtered_hemisphere_colors)
plt.title('Degree Vs Closeness')
plt.xlabel('Degree_Centrality')
plt.ylabel('Closeness_Centrality')
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for label, color in hemisphere_color_map.items()])
plt.savefig('multiplex_scatter.png')
plt.show()


plt.scatter(node_data['PC1'], node_data['PC2'], c=filtered_hemisphere_colors)
plt.title('Nodes measures PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for label, color in hemisphere_color_map.items()])
plt.savefig('multiplex_multiPCA.png')
plt.show()

couple = set()

for i in range(1609):
    if node_data['PC1'][i] > 0.3:
        node_id = node_data['Node'][i]
        couple.add(node_id)

print("Nodi nel set 'couple':", couple)
for node_id in couple:
    if multiplex.has_node(node_id):
        node_attributes = multiplex.nodes[node_id]
        print(f"Node ID: {node_id}, Attributes: {node_attributes}")

adjacency_matrix = nx.adjacency_matrix(multiplex)
pca = PCA(n_components = 2)  
principal_components = pca.fit_transform(adjacency_matrix.toarray())
node_data['PC1'] = principal_components[:, 0]
node_data['PC2'] = principal_components[:, 1]
print(pca.explained_variance_ratio_)

particular_nodes = set()
for i in range(1609):
    if node_data['PC1'][i] >= 10 and abs(node_data['PC2'][i]) < 10.5 :
        node_id = node_data['Node'][i]
        particular_nodes.add(node_id)
print('particular nodes')
for node_id in particular_nodes:
    if multiplex.has_node(node_id):
        node_attributes = multiplex.nodes[node_id]
        print(f"Node ID: {node_id}, Attributes: {node_attributes}")
subgraph = multiplex.subgraph(particular_nodes)
undirected_graph = nx.Graph(subgraph)

is_connected = nx.is_connected(undirected_graph)

if is_connected:
    print("Il sottoinsieme di nodi è connesso.")
else:
    print("Il sottoinsieme di nodi non è connesso.")      

plt.scatter(principal_components[:, 0], principal_components[:, 1], c=filtered_hemisphere_colors)
plt.title('Adjacency matrix PCA ')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for label, color in hemisphere_color_map.items()])
plt.savefig('multiplex_PCA.png')
plt.show()

node_colors = [hemisphere_color_map.get(subgraph.nodes[node].get('Hemisphere', 'nan')) for node in subgraph.nodes]

left_nodes = [node for node in subgraph.nodes if subgraph.nodes[node].get('Hemisphere', 'nan') == 'left']
right_nodes = [node for node in subgraph.nodes if subgraph.nodes[node].get('Hemisphere', 'nan') == 'right']

pos_left = nx.spring_layout(subgraph.subgraph(left_nodes), seed=32)
pos_right = nx.spring_layout(subgraph.subgraph(right_nodes), seed=42)



pos = {**pos_left, **pos_right}
#pos = nx.kamada_kawai_layout(subgraph)
nx.draw(subgraph, pos, with_labels=True, node_size=700, node_color=node_colors , font_size=10, font_color="black", font_weight="bold", edge_color="gray", linewidths=1, alpha=0.7)
plt.savefig('particular_nodes_graph.png')
plt.show()

