import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

data_file = pd.read_csv('nodes.csv')
counts_hemisphere = data_file[' hemisphere'].value_counts()
fraction_hemisphere = counts_hemisphere * 100 / data_file.shape[0] 
print(counts_hemisphere)
print(fraction_hemisphere)