import numpy as np
import networkx as nx
from scipy.io import mmread

# ---------- LOAD ERDOS (.mtx) ----------
MTX_PATH = "ca-Erdos992.mtx"

print("Loading Erdos graph...")
A = mmread(MTX_PATH).tocsr()
A.data[:] = 1
G = nx.from_scipy_sparse_array(A)   

print(f"Loaded graph: n={G.number_of_nodes()}, m={G.number_of_edges()}")

# Use largest connected component for betweenness 
largest_cc_nodes = max(nx.connected_components(G), key=len)
Gcc = G.subgraph(largest_cc_nodes).copy()
print(f"Largest CC size = {Gcc.number_of_nodes()}")

# ---------- CENTRALITIES ----------
print("\nComputing degree centrality (fast)...")
deg_cent = nx.degree_centrality(Gcc)  # use GCC for consistent node set

print("Computing betweenness centrality (approx)...")
# If it's slow, reduce k to 50 or 100
k = min(200, Gcc.number_of_nodes())
btw_cent = nx.betweenness_centrality(Gcc, k=k, seed=0, normalized=True)

# ---------- TOP 10% SETS ----------
def top_percent_set(metric_dict, percent=0.10):
    """Return the set of nodes in the top 'percent' by metric value."""
    values = np.array(list(metric_dict.values()))
    threshold = np.quantile(values, 1 - percent)  # 90th percentile for top 10%
    return {v for v, s in metric_dict.items() if s >= threshold}

top10_deg = top_percent_set(deg_cent, 0.10)
top10_btw = top_percent_set(btw_cent, 0.10)

overlap = top10_deg & top10_btw

print("\n--- Top 10% overlap (Degree AND Betweenness) ---")
print(f"Top 10% by degree:       {len(top10_deg)} nodes")
print(f"Top 10% by betweenness:  {len(top10_btw)} nodes")
print(f"Overlap (both):          {len(overlap)} nodes")
print(f"Overlap / top-degree set = {len(overlap)/len(top10_deg):.3f}")
print(f"Overlap / top-btw set     = {len(overlap)/len(top10_btw):.3f}")

# Optional: save node lists to text files 
with open("erdos_top10_degree_nodes.txt", "w") as f:
    for v in sorted(top10_deg):
        f.write(f"{v}\n")

with open("erdos_top10_betweenness_nodes.txt", "w") as f:
    for v in sorted(top10_btw):
        f.write(f"{v}\n")

with open("erdos_top10_overlap_nodes.txt", "w") as f:
    for v in sorted(overlap):
        f.write(f"{v}\n")

print("\nSaved node lists: erdos_top10_degree_nodes.txt, erdos_top10_betweenness_nodes.txt, erdos_top10_overlap_nodes.txt")
