import numpy as np
import networkx as nx
from scipy.io import mmread
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter

# --------- Load Erdos (largest connected component) ----------
MTX_PATH = "ca-Erdos992.mtx"

print("Loading graph...")
A = mmread(MTX_PATH).tocsr()
A.data[:] = 1
G = nx.from_scipy_sparse_array(A)

gcc_nodes = max(nx.connected_components(G), key=len)
Gcc = G.subgraph(gcc_nodes).copy()
print(f"GCC: n={Gcc.number_of_nodes()}, m={Gcc.number_of_edges()}")

# Utility: save partition + summarize
def summarize_partition(part, title):
    sizes = Counter(part.values())
    print(f"\n[{title}]")
    print("  #communities:", len(sizes))
    print("  largest 10 sizes:", sizes.most_common(10))

def export_gexf_with_attr(G, part, filename, attr_name):
    nx.set_node_attributes(G, part, attr_name)
    nx.write_gexf(G, filename)
    print(f"  saved: {filename}  (node attribute: {attr_name})")

# --------- 1) Louvain (UNBOUNDED) ----------
import community as community_louvain

print("\nRunning Louvain (unbounded)...")
part_louvain = community_louvain.best_partition(Gcc)  
summarize_partition(part_louvain, "Louvain (unbounded)")
export_gexf_with_attr(Gcc, part_louvain, "erdos_louvain.gexf", "community_louvain")

# --------- 2) Spectral clustering (BOUNDED by choosing k) ----------
def spectral_partition(G, k, seed=0):
    L = nx.normalized_laplacian_matrix(G).astype(float)

    # Compute k smallest eigenvectors of L
    vals, vecs = eigsh(L, k=k, which="SM")
    X = vecs  # n x k embedding

    # KMeans to get k clusters (bounded)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(X)

    return {node: int(lbl) for node, lbl in zip(G.nodes(), labels)}

for k in [5, 10, 20]:
    print(f"\nRunning Spectral clustering (bounded, k={k})...")
    part_spec = spectral_partition(Gcc, k=k, seed=0)
    summarize_partition(part_spec, f"Spectral (k={k}) [bounded]")
    export_gexf_with_attr(Gcc, part_spec, f"erdos_spectral_k{k}.gexf", f"community_spec_k{k}")

# --------- 3) Quick visualization in Python (sampled subgraph) ----------
def plot_sample(G, part, title, out_png, sample_n=800, seed=0):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    if len(nodes) > sample_n:
        keep = set(rng.choice(nodes, size=sample_n, replace=False))
        H = G.subgraph(keep).copy()
    else:
        H = G.copy()

    # layout
    pos = nx.spring_layout(H, seed=seed)

    # colors by community id
    comm = np.array([part[v] for v in H.nodes()])
    plt.figure(figsize=(9, 7))
    nx.draw_networkx_nodes(H, pos, node_size=12, node_color=comm, cmap="tab20")
    nx.draw_networkx_edges(H, pos, alpha=0.15, width=0.6)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"  saved plot: {out_png}")

# plot Louvain + one spectral example
plot_sample(Gcc, part_louvain, "Erdos GCC communities (Louvain)", "plot_erdos_louvain.png", sample_n=800, seed=0)

# reuse the last spectral partition (k=20) for a plot
part_spec20 = spectral_partition(Gcc, k=20, seed=0)
plot_sample(Gcc, part_spec20, "Erdos GCC communities (Spectral, k=20)", "plot_erdos_spectral_k20.png", sample_n=800, seed=0)

print("\nDone. Open .gexf in Gephi for full visualization.")
