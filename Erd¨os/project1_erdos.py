import numpy as np
import networkx as nx
from scipy.io import mmread

MTX_PATH = "ca-Erdos992.mtx"  

A = mmread(MTX_PATH).tocsr()
A.data[:] = 1

G = nx.from_scipy_sparse_array(A)

print("Loaded graph:")
print("n (nodes) =", G.number_of_nodes())
print("m (edges) =", G.number_of_edges())

largest_cc_nodes = max(nx.connected_components(G), key=len)
Gcc = G.subgraph(largest_cc_nodes).copy()
print("largest CC size =", Gcc.number_of_nodes())
