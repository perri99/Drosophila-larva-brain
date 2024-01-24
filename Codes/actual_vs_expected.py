import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import itertools
import pickle
from scipy.stats import chi2_contingency
'''
MUST BE EXECUTED AFTER actual_edges.py and expected_edge.py
This program compare the actual number edges vs expected number of edgese
The input files are the output files of actual_edges.py and expected_edge.py
'''
def normalizza_chiave(chiave):
    return '+'.join(sorted(chiave.split('+')))

with open('expected_edges.pkl', 'rb') as file:
    expected_edges = pickle.load(file)

with open('actual_edges.pkl', 'rb') as file:
    actual_edges = pickle.load(file)





expected_edges_norm = {normalizza_chiave(k): v for k, v in expected_edges.items()}
actual_edges_norm = {normalizza_chiave(k): v for k, v in actual_edges.items()}


# ordered list of keys
key_order = sorted(set(expected_edges_norm.keys()) | set(actual_edges_norm.keys()))

#contingency table for chi2 test
contingency_table = [[expected_edges_norm.get(k, 0), actual_edges_norm.get(k, 0)] for k in key_order]

# ch2 test
chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)

print(f"Chi2 Stat: {chi2_stat}")
print(f"P-value del test del chi-quadro: {p_value}")

total_edges = sum(actual_edges_norm.values())
print(f'Total edges= {total_edges}')

one_type_edges = actual_edges['aa'] + actual_edges['ad'] + actual_edges['da'] + actual_edges['dd']
print(f'One type edges = {one_type_edges}')
print(one_type_edges/total_edges)
key_order = list(actual_edges_norm.keys())

expected_edges1 = {k: expected_edges_norm[k] for k in key_order}
actual_edges1 = {k: actual_edges_norm[k] for k in key_order}
  
etypes = list(actual_edges1.keys())
expected_values = list(expected_edges1.values())
actual_values = list(actual_edges1.values())

bar_width = 0.35  # Width of the bars
index = np.arange(len(etypes))  # Index for the x-axis

# Plotting
plt.bar(index, expected_values, width=bar_width, label='Expected Links')
plt.bar(index + bar_width, actual_values, width=bar_width, label='Actual Links')

# Customizing the plot
plt.xlabel('Etype')
plt.ylabel('Links')
plt.yscale('log')
plt.ylim(1,10**5)
plt.title('Expected and Actual Links for Each Etype')
plt.xticks(index + bar_width / 2, etypes)  # Positioning x-axis labels
plt.legend()
plt.show()
