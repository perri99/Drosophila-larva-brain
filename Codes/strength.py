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

# Ottieni l'insieme completo dei nodi del multigrafo
all_nodes = set(multigraph.nodes())
subgraphs = [multigraph.edge_subgraph(((u, v, k) for u, v, k, data in multigraph.edges(keys=True, data=True) if data['attribute'] == etype)) for etype in set(nx.get_edge_attributes(multigraph, 'attribute').values())]

link_counts = {}
synapses_counts = {}
total = {}
# Inizializza link_counts per ciascun attributo
for graph in subgraphs:
    for _, _, data in graph.edges(data=True):
        etype = data['attribute']
        if etype not in link_counts:
            link_counts[etype] = [0, 0, 0, 0, 0]
            synapses_counts[etype] = [0, 0, 0, 0, 0]

# Conta i link con diversi pesi
for graph in subgraphs:
    total_links = graph.number_of_edges()
    total_synapses = sum(data['weight'] for _, _, data in graph.edges(data=True))
    for _, _, data in graph.edges(data=True):
        etype = data['attribute']
        weight = data['weight']
        
        if weight == 1:
            link_counts[etype][0] += 1 / total_links
            synapses_counts[etype][0] += 1 / total_synapses
        elif weight == 2:
            link_counts[etype][1] += 1 / total_links
            synapses_counts[etype][1] += 2 / total_synapses
        elif weight == 3:
            link_counts[etype][2] += 1 / total_links
            synapses_counts[etype][2] += 3 / total_synapses
        elif weight == 4:
            link_counts[etype][3] += 1 / total_links
            synapses_counts[etype][3] += 4 / total_synapses
        elif weight > 4:
            link_counts[etype][4] += 1 / total_links
            synapses_counts[etype][4] += weight / total_synapses
    total[etype] = synapses_counts[etype][0] + synapses_counts[etype][1] + synapses_counts[etype][2] + synapses_counts[etype][3] +synapses_counts[etype][4]
print(total)
#etype_histo = list(link_counts.keys())
#counts = list(link_counts.values())
x_axis = np.arange(1, 6)  # Pesos da 1 a 5

bar_width = 0.2  # Larghezza delle barre

# Crea un grafico a barre per ogni etichetta (etype)
for i, (etype, counts) in enumerate(link_counts.items()):
    plt.bar(x_axis + i * bar_width, counts, width=bar_width, label=etype)
# Aggiungi titoli e etichette
plt.title('Edges distribution by synaptic strength')
plt.xlabel('Weights')
plt.ylabel('Edges fraction')
plt.legend()

# Mostra il grafico
plt.savefig('Plots/Edges distribution by strength.png')
plt.clf()
# Crea un grafico a barre per ogni etichetta (etype)
for i, (etype, counts) in enumerate(synapses_counts.items()):
    plt.bar(x_axis + i * bar_width, counts, width=bar_width, label=etype)
# Aggiungi titoli e etichette
plt.title('Synapses in edges of different strength')
plt.xlabel('Weights')
plt.ylabel('Synapses fraction')
plt.legend()

# Mostra il grafico
plt.savefig('Plots/Synapses distribution by strength.png')