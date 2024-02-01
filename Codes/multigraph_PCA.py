import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
import community  

# reading edges
edge_list_path = 'edges.csv'  # Sostituisci con il percorso effettivo del tuo file
df = pd.read_csv(edge_list_path)
nodes_path = 'nodes.csv'
read_nodes = pd.read_csv(nodes_path)

multigraph = nx.MultiDiGraph()

# linking construction
for _, row in df.iterrows():
    source, target, weight, etype = row['source'], row['target'], row['weight'], row['etype']
    multigraph.add_edge(source, target, weight=weight, attribute=etype)
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



#print(list(multigraph.nodes(data = True)))
density = nx.density(multigraph)
degree_sequence = sorted((d for n, d in multigraph.degree()), reverse=True)
max_degree = max(degree_sequence)
nodes_number = nx.number_of_nodes(multigraph)
synapses = sum(weight['weight'] for _, _, weight in multigraph.edges(data=True))
print(f' Number of nodes = {nodes_number} Number of edges = {multigraph.number_of_edges()}  Number of synapses = {synapses} Density = {density} Max Degree = {max_degree}')

degree_centrality = nx.degree_centrality(multigraph)
closeness_centrality = nx.closeness_centrality(multigraph)
betweenness_centrality = nx.betweenness_centrality(multigraph)
#eigenvector_centrality = nx.eigenvector_centrality(multigraph)
pagerank = nx.pagerank(multigraph)

node_data = pd.DataFrame({
    'Node': list(multigraph.nodes()),
    'Degree_Centrality': list(degree_centrality.values()),
    'Closeness_Centrality': list(closeness_centrality.values()),
    'Betweenness_Centrality': list(betweenness_centrality.values()),
    'PageRank': list(pagerank.values())
})


features = ['Degree_Centrality', 'Closeness_Centrality', 'Betweenness_Centrality', 'PageRank']
X = node_data[features]

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
print(pca.explained_variance_ratio_)

node_data['PC1'] = principal_components[:, 0]
node_data['PC2'] = principal_components[:, 1]
hemisphere_color_map = {'left': 'blue', 'right': 'red', 'nan': 'green'}

filtered_hemisphere_colors = []
default_color = 'gray'  

for node in multigraph.nodes:
    hemisphere = multigraph.nodes[node].get('Hemisphere', 'nan')
    color = hemisphere_color_map.get(hemisphere, default_color)
    filtered_hemisphere_colors.append(color)

plt.scatter(node_data['Degree_Centrality'], node_data['Closeness_Centrality'], c=filtered_hemisphere_colors)
plt.title('Degree vs Closeness')
plt.xlabel('Degree centrality')
plt.ylabel('Closeness centrality')
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for label, color in hemisphere_color_map.items()])
plt.savefig('multigraph_multiPCA.png')
plt.show()


plt.scatter(node_data['PC1'], node_data['PC2'], c=filtered_hemisphere_colors)
plt.title('PCA delle Misure dei Nodi nel multigraph Network')
plt.xlabel('Degree centrality')
plt.ylabel('Closeness centrality')
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for label, color in hemisphere_color_map.items()])
plt.savefig('multigraph_multiPCA.png')
plt.show()

couple = set()

for i in range(2952):
    if node_data['PC1'][i] > 0.3:
        node_id = node_data['Node'][i]
        couple.add(node_id)


print("Nodi nel set 'couple':", couple)
for node_id in couple:
    if multigraph.has_node(node_id):
        node_attributes = multigraph.nodes[node_id]
        print(f"Node ID: {node_id}, Attributes: {node_attributes}")

adjacency_matrix = nx.adjacency_matrix(multigraph)
pca = PCA()  
principal_components = pca.fit_transform(adjacency_matrix.toarray())
node_data['PC1'] = principal_components[:, 0]
node_data['PC2'] = principal_components[:, 1]
print(pca.explained_variance_ratio_)
# Grafico PCA del multigraph Network
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=filtered_hemisphere_colors)
plt.title('PCA of multigraph Network')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for label, color in hemisphere_color_map.items()])
plt.savefig('multigraph_PCA.png')
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))

axes[0].bar(node_data['Node'], node_data['PC1'], color = 'blue' )
axes[0].set_title('PCA of multigraph Network - PC1')
axes[0].set_xlabel('Node')
axes[0].set_ylabel('Principal Component 1')


axes[1].bar(node_data['Node'], node_data['PC2'], color = 'blue')
axes[1].set_title('PCA of multigraph Network - PC2')
axes[1].set_xlabel('Node')
axes[1].set_ylabel('Principal Component 2')

plt.tight_layout() 
plt.savefig('multigraph_PCA_combined.png')
plt.show()
