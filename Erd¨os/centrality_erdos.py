import numpy as np
import networkx as nx
from scipy.io import mmread
from collections import Counter

MTX_PATH = "ca-Erdos992.mtx"   

def topk(d, k=10):
    """Return top-k (node, score) pairs sorted by score."""
    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]

print("Loading graph...")
A = mmread(MTX_PATH).tocsr()
A.data[:] = 1
G = nx.from_scipy_sparse_array(A)  
print(f"Loaded graph: n={G.number_of_nodes()}, m={G.number_of_edges()}")

# Largest connected component 
largest_cc_nodes = max(nx.connected_components(G), key=len)
Gcc = G.subgraph(largest_cc_nodes).copy()
print(f"Largest CC size = {Gcc.number_of_nodes()}")

# -------------------------
# 1) Degree centrality (fast)
# -------------------------
print("\nComputing degree centrality...")
deg_cent = nx.degree_centrality(G)  # uses whole graph

# Also get raw degrees for comparison on whole graph
deg = dict(G.degree())

print("Top 10 by degree (raw):", topk(deg, 10))
print("Top 10 by degree centrality:", topk(deg_cent, 10))

# -------------------------
# 2) Betweenness centrality (approx)
# -------------------------
print("\nComputing betweenness centrality (approx)...")
k = min(200, Gcc.number_of_nodes())
btw_cent = nx.betweenness_centrality(Gcc, k=k, seed=0, normalized=True)

print("Top 10 by approx betweenness:", topk(btw_cent, 10))

# -------------------------
# 3) Eigenvector centrality (iterative)
# -------------------------
print("\nComputing eigenvector centrality...")
eig_cent = nx.eigenvector_centrality(Gcc, max_iter=3000, tol=1e-06)

print("Top 10 by eigenvector:", topk(eig_cent, 10))

# -------------------------
# Save results to a CSV 
# -------------------------
import pandas as pd

# Create one table of node scores 
nodes = list(Gcc.nodes())
df = pd.DataFrame({
    "node": nodes,
    "degree": [deg.get(v, 0) for v in nodes],
    "degree_centrality": [deg_cent.get(v, 0.0) for v in nodes],
    "betweenness_centrality": [btw_cent.get(v, 0.0) for v in nodes],
    "eigenvector_centrality": [eig_cent.get(v, 0.0) for v in nodes],
})

df.to_csv("erdos_centralities_gcc.csv", index=False)
print("\nSaved: erdos_centralities_gcc.csv")

# -------------------------
# Quick overlap check: are the same nodes central in multiple metrics?
# -------------------------
def top_percent_set(metric_dict, pct=0.01):
    # top 1% by default
    n = max(1, int(len(metric_dict) * pct))
    return {v for v, _ in topk(metric_dict, n)}

top_deg_1 = top_percent_set({v: deg_cent[v] for v in Gcc.nodes() if v in deg_cent}, 0.01)
top_btw_1 = top_percent_set(btw_cent, 0.01)
top_eig_1 = top_percent_set(eig_cent, 0.01)

print("\nTop 1% overlap sizes:")
print("deg ∩ btw =", len(top_deg_1 & top_btw_1))
print("deg ∩ eig =", len(top_deg_1 & top_eig_1))
print("btw ∩ eig =", len(top_btw_1 & top_eig_1))
