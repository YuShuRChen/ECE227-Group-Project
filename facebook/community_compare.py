import networkx as nx
import matplotlib.pyplot as plt

# Louvain (unbounded)
import community as community_louvain

# Bounded community detection
from sklearn.cluster import SpectralClustering
import numpy as np


def load_facebook_graph(path="facebook_combined.txt"):
    G = nx.read_edgelist(path, nodetype=int)
    return G


def louvain_partition(G, seed=42):
    # returns dict: node -> community_id
    part = community_louvain.best_partition(G, random_state=seed)
    return part


def bounded_spectral_partition(G, k=10, seed=42):
    """
    Bounded approach: fix number of communities = k.
    Use spectral clustering on adjacency matrix.
    Returns dict: node -> cluster_id

    Note: This requires building an N x N adjacency matrix; for N=4039 it's OK.
    """
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # adjacency matrix (dense) for simplicity
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=float)

    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        random_state=seed,
        assign_labels="kmeans",
    )
    labels = sc.fit_predict(A)

    part = {nodes[i]: int(labels[i]) for i in range(len(nodes))}
    return part


def summarize_partition(G, part, name="partition"):
    # number of communities and top sizes
    comm_ids = list(set(part.values()))
    sizes = {}
    for n, c in part.items():
        sizes[c] = sizes.get(c, 0) + 1
    sizes_sorted = sorted(sizes.items(), key=lambda x: x[1], reverse=True)

    print(f"\n===== {name} =====")
    print("Communities:", len(comm_ids))
    print("Top 10 community sizes:", sizes_sorted[:10])

    # modularity (well-defined for any partition; higher typically better)
    modularity = community_louvain.modularity(part, G)
    print("Modularity:", modularity)

    return len(comm_ids), sizes_sorted, modularity


def draw_partition(G, part, title, layout_pos=None, node_size=8):
    """
    Visualize communities by coloring nodes.
    For fair comparison, pass the same layout_pos to both plots.
    """
    if layout_pos is None:
        # spring layout can take a bit; set seed for repeatability
        layout_pos = nx.spring_layout(G, seed=42, k=None)

    # map community id to consecutive color indices
    comms = sorted(set(part.values()))
    comm_to_color = {c: i for i, c in enumerate(comms)}
    node_colors = [comm_to_color[part[n]] for n in G.nodes()]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(G, layout_pos, alpha=0.05, width=0.5)
    nx.draw_networkx_nodes(G, layout_pos, node_color=node_colors, node_size=node_size)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    return layout_pos


def overlap_between_partitions(partA, partB):
    """
    Not required, but helpful: compute simple agreement rate
    (note labels are arbitrary, so raw label match isn't meaningful).
    Instead, we just report:
    - how many communities in each
    - modularity in each
    """
    pass


def main():
    G = load_facebook_graph("facebook_combined.txt")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("Connected:", nx.is_connected(G))

    # Optional: use largest connected component for cleaner community detection
    # (Facebook combined is connected in SNAP, but we keep this robust)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print("Using largest connected component:")
        print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

    # 1) Unbounded: Louvain
    part_louvain = louvain_partition(G, seed=42)
    summarize_partition(G, part_louvain, name="Louvain (unbounded)")

    # 2) Bounded: Spectral clustering with fixed k
    k = 10  # you can change this (e.g., 8, 12, 15) and compare modularity
    part_bounded = bounded_spectral_partition(G, k=k, seed=42)
    summarize_partition(G, part_bounded, name=f"Spectral Clustering (bounded, k={k})")

    # Use same layout for fair visual comparison
    pos = nx.spring_layout(G, seed=42)

    draw_partition(G, part_louvain, "Facebook Communities - Louvain (unbounded)", layout_pos=pos, node_size=10)
    plt.savefig("facebook_louvain.png", dpi=200)

    draw_partition(G, part_bounded, f"Facebook Communities - Spectral (bounded, k={k})", layout_pos=pos, node_size=10)
    plt.savefig("facebook_bounded_spectral.png", dpi=200)

    print("\nSaved figures:")
    print("- facebook_louvain.png")
    print(f"- facebook_bounded_spectral.png")


if __name__ == "__main__":
    main()

