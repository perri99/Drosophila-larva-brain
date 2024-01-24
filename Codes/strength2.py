import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp


# reading edges
edge_list_path = 'edges.csv'  # Sostituisci con il percorso effettivo del tuo file
df = pd.read_csv(edge_list_path)

multigraph = nx.MultiDiGraph()

# linking construction
for _, row in df.iterrows():
    source, target, weight, etype = row['source'], row['target'], row['weight'], row['etype']
    multigraph.add_edge(source, target, weight=weight, attribute=etype)

unique_etypes = set(data['attribute'] for _, _, data in multigraph.edges(data=True))
weights_dict = {etype: {'weight_5': 0, 'weight_1_2': 0} for etype in unique_etypes}
synapses = {etype: {'weight_5': 0, 'weight_1_2': 0} for etype in unique_etypes}

subgraphs = [multigraph.edge_subgraph(((u, v, k) for u, v, k, data in multigraph.edges(keys=True, data=True) if data['attribute'] == etype)) for etype in set(nx.get_edge_attributes(multigraph, 'attribute').values())]
total_links = {}
total_synapses = {}
for graph in subgraphs:
    for _, _, data in graph.edges(data=True):
        etype = data['attribute']
    total_links[etype] = graph.number_of_edges()
    total_synapses[etype] = sum(data['weight'] for _, _, data in graph.edges(data=True))
# Calcola la somma dei pesi per ciascun etype e peso specifico
for _, _, data in multigraph.edges(data=True):
    etype = data['attribute']
    weight = data['weight']
    if weight > 4:
        weights_dict[etype]['weight_5'] += 1 /total_links[etype]
        synapses[etype]['weight_5'] += weight / total_synapses[etype]
    elif weight in [1, 2]:
        weights_dict[etype]['weight_1_2'] += 1 /total_links[etype]
        synapses[etype]['weight_1_2'] += weight / total_synapses[etype]


# Prepara dati per il grafico a barre
etypenames = list(unique_etypes)
weight_5_values = [weights_dict[etype]['weight_5'] for etype in etypenames]
weight_1_2_values = [weights_dict[etype]['weight_1_2'] for etype in etypenames]
synapses_5_values = [synapses[etype]['weight_5'] for etype in etypenames]
synapses_1_2_values = [synapses[etype]['weight_1_2'] for etype in etypenames]
# Crea il grafico a barre
bar_width = 0.35
index = range(len(etypenames))

fig, ax = plt.subplots()
bar1 = ax.bar(index, weight_5_values, bar_width, label='Weight >= 5')
bar2 = ax.bar([i + bar_width for i in index], weight_1_2_values, bar_width, label='Weight 1+2')

# Aggiungi dettagli al grafico
ax.set_xlabel('Link type')
ax.set_ylabel('Fraction of edges')
ax.set_title('Edges for etype')
ax.set_xticks([i + bar_width/2 for i in index])
ax.set_xticklabels(etypenames)
ax.legend()

# Mostra il grafico
plt.savefig("Plots/Weak and strong edges fraction.png")

fig, ax = plt.subplots()
bar1 = ax.bar(index,synapses_5_values, bar_width, label='Weight >= 5')
bar2 = ax.bar([i + bar_width for i in index], synapses_1_2_values, bar_width, label='Weight 1+2')

# Aggiungi dettagli al grafico
ax.set_xlabel('Link type')
ax.set_ylabel('Fraction of synapses')
ax.set_title('Synapses for etype')
ax.set_xticks([i + bar_width/2 for i in index])
ax.set_xticklabels(etypenames)
ax.legend()

# Mostra il grafico
plt.savefig("Plots/Weak and strong synapses fraction.png")