import networkx as nx
from scipy.io import mmread
from collections import Counter, defaultdict
import pandas as pd
import community as community_louvain

MTX_PATH = "ca-Erdos992.mtx"

# Load graph + largest connected component
A = mmread(MTX_PATH).tocsr()
A.data[:] = 1
G = nx.from_scipy_sparse_array(A)

gcc_nodes = max(nx.connected_components(G), key=len)
Gcc = G.subgraph(gcc_nodes).copy()

# Louvain partition (unbounded)
part = community_louvain.best_partition(Gcc)

# Degree (raw) on GCC
deg = dict(Gcc.degree())

# Find largest communities
comm_sizes = Counter(part.values())
largest5 = [cid for cid, _ in comm_sizes.most_common(5)]
print("Largest 5 communities (id, size):", comm_sizes.most_common(5))

rows = []
for cid in largest5:
    nodes_in_c = [v for v in Gcc.nodes() if part[v] == cid]
    # sort by degree descending
    top5 = sorted(nodes_in_c, key=lambda v: deg[v], reverse=True)[:5]
    for rank, v in enumerate(top5, start=1):
        rows.append({
            "community_id": cid,
            "community_size": comm_sizes[cid],
            "rank_in_comm": rank,
            "node_id": v,
            "degree": deg[v],
        })

df = pd.DataFrame(rows)
print(df)

# Save for report
df.to_csv("erdos_louvain_top5_by_degree_in_largest5_communities.csv", index=False)
print("\nSaved: erdos_louvain_top5_by_degree_in_largest5_communities.csv")
