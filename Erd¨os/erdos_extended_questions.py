import random
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from scipy.io import mmread


# ----------------------------
# Load graph from .mtx
# ----------------------------
def load_graph_from_mtx(mtx_path: str) -> nx.Graph:
    A = mmread(mtx_path).tocsr()

    A.data = np.ones_like(A.data)

    A.setdiag(0)
    A.eliminate_zeros()

    G = nx.from_scipy_sparse_array(A)

    if isinstance(G, nx.MultiGraph):
        G = nx.Graph(G)

    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def get_gcc_subgraph(G: nx.Graph) -> nx.Graph:
    """Return a copy of the Giant Connected Component subgraph."""
    if G.number_of_nodes() == 0:
        return G.copy()
    gcc_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(gcc_nodes).copy()


# ----------------------------
# Metrics
# ----------------------------
def gcc_fraction(G: nx.Graph) -> float:
    """Fraction of nodes in the largest connected component."""
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    return len(max(nx.connected_components(G), key=len)) / n


def approx_gcc_aspl(G: nx.Graph, sample_sources: int = 40, seed: int = 0) -> float:
    """
    Fast approximate average shortest path length (ASPL) in the GCC.
    Method: sample `sample_sources` nodes in the GCC, run BFS from each, average distances.

    - Much faster than nx.average_shortest_path_length(H)
    - Returns NaN if GCC has < 2 nodes
    """
    if G.number_of_nodes() < 2:
        return float("nan")

    gcc_nodes = list(max(nx.connected_components(G), key=len))
    if len(gcc_nodes) < 2:
        return float("nan")

    H = G.subgraph(gcc_nodes)

    rng = random.Random(seed)
    k = min(sample_sources, len(gcc_nodes))
    sources = rng.sample(gcc_nodes, k=k)

    dists = []
    for s in sources:
        lengths = nx.single_source_shortest_path_length(H, s)
        dists.extend(lengths.values())

    if len(dists) == 0:
        return float("nan")
    return float(np.mean(dists))


# ----------------------------
# Removal strategies
# ----------------------------
def removal_order_random(G: nx.Graph, seed: int) -> list:
    rng = random.Random(seed)
    order = list(G.nodes())
    rng.shuffle(order)
    return order


def removal_order_degree(G: nx.Graph) -> list:
    return [v for v, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)]


def removal_order_coreness(G: nx.Graph) -> list:
    core = nx.core_number(G)
    return [v for v, _ in sorted(core.items(), key=lambda x: x[1], reverse=True)]


def simulate_progressive_removal(
    G: nx.Graph,
    order: list,
    fractions: np.ndarray,
    aspl_samples: int = 40,
    aspl_seed: int = 0,
) -> dict:
    """
    Remove nodes progressively according to `order`.
    At each fraction f, remove round(f * N0) nodes total, then record:
      - GCC fraction
      - number of connected components
      - approx ASPL in GCC
    """
    Gwork = G.copy()
    n0 = Gwork.number_of_nodes()
    if n0 == 0:
        raise ValueError("Graph has no nodes.")

    removed = 0
    out = {
        "fraction_removed": [],
        "gcc_fraction": [],
        "num_components": [],
        "gcc_aspl": [],
    }

    for idx, f in enumerate(fractions):
        target = int(round(f * n0))
        while removed < target and removed < len(order) and Gwork.number_of_nodes() > 0:
            v = order[removed]
            if Gwork.has_node(v):
                Gwork.remove_node(v)
            removed += 1

        out["fraction_removed"].append(f)

        if Gwork.number_of_nodes() == 0:
            out["gcc_fraction"].append(0.0)
            out["num_components"].append(0)
            out["gcc_aspl"].append(float("nan"))
            continue

        out["gcc_fraction"].append(gcc_fraction(Gwork))
        out["num_components"].append(nx.number_connected_components(Gwork))

        # Make ASPL deterministic per step by varying seed with idx
        out["gcc_aspl"].append(
            approx_gcc_aspl(Gwork, sample_sources=aspl_samples, seed=aspl_seed + idx)
        )

    # arrays
    out["fraction_removed"] = np.array(out["fraction_removed"], dtype=float)
    out["gcc_fraction"] = np.array(out["gcc_fraction"], dtype=float)
    out["num_components"] = np.array(out["num_components"], dtype=int)
    out["gcc_aspl"] = np.array(out["gcc_aspl"], dtype=float)
    return out


def run_random_trials(
    G: nx.Graph,
    fractions: np.ndarray,
    trials: int = 10,
    seed0: int = 0,
    aspl_samples: int = 40,
) -> dict:
    """
    Multiple random removal trials -> mean and std of:
      - GCC fraction
      - approx ASPL in GCC
    """
    gcc_mat = []
    aspl_mat = []

    for t in range(trials):
        order = removal_order_random(G, seed=seed0 + t)
        res = simulate_progressive_removal(
            G,
            order,
            fractions,
            aspl_samples=aspl_samples,
            aspl_seed=1000 + 17 * t,  # different deterministic seed per trial
        )
        gcc_mat.append(res["gcc_fraction"])
        aspl_mat.append(res["gcc_aspl"])

    gcc_mat = np.vstack(gcc_mat)
    aspl_mat = np.vstack(aspl_mat)

    # Avoid warnings if some entries become NaN when GCC tiny
    with np.errstate(all="ignore"):
        gcc_mean = np.nanmean(gcc_mat, axis=0)
        gcc_std = np.nanstd(gcc_mat, axis=0)
        aspl_mean = np.nanmean(aspl_mat, axis=0)
        aspl_std = np.nanstd(aspl_mat, axis=0)

    return {
        "fraction_removed": fractions,
        "gcc_mean": gcc_mean,
        "gcc_std": gcc_std,
        "aspl_mean": aspl_mean,
        "aspl_std": aspl_std,
    }


# ----------------------------
# Plot helpers
# ----------------------------
def plot_gcc_curves(fractions, random_stats, deg_res, core_res=None, title=""):
    plt.figure()
    plt.plot(fractions, random_stats["gcc_mean"], marker="o", linestyle="-", label="Random (mean)")
    plt.fill_between(
        fractions,
        np.maximum(0, random_stats["gcc_mean"] - random_stats["gcc_std"]),
        np.minimum(1, random_stats["gcc_mean"] + random_stats["gcc_std"]),
        alpha=0.2,
        label="Random (±1 std)",
    )
    plt.plot(fractions, deg_res["gcc_fraction"], marker="o", linestyle="-", label="Targeted: degree")
    if core_res is not None:
        plt.plot(fractions, core_res["gcc_fraction"], marker="o", linestyle="-", label="Targeted: coreness (k-core)")

    plt.xlabel("Fraction of nodes removed")
    plt.ylabel("Fraction in largest connected component (GCC)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_aspl_curves(fractions, random_stats, deg_res, core_res=None, title=""):
    plt.figure()
    plt.plot(fractions, random_stats["aspl_mean"], marker="o", linestyle="-", label="Random (mean)")
    plt.fill_between(
        fractions,
        random_stats["aspl_mean"] - random_stats["aspl_std"],
        random_stats["aspl_mean"] + random_stats["aspl_std"],
        alpha=0.2,
        label="Random (±1 std)",
    )
    plt.plot(fractions, deg_res["gcc_aspl"], marker="o", linestyle="-", label="Targeted: degree")
    if core_res is not None:
        plt.plot(fractions, core_res["gcc_aspl"], marker="o", linestyle="-", label="Targeted: coreness (k-core)")

    plt.xlabel("Fraction of nodes removed")
    plt.ylabel("Approx avg shortest path length in GCC (sampled)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    mtx_path = "ca-Erdos992.mtx"

    # Strong defaults for your dataset:
    USE_GCC_ONLY = True
    MAX_REMOVE = 0.25      
    STEPS = 21             
    RANDOM_TRIALS = 8      
    ASPL_SAMPLES = 35      

    G0 = load_graph_from_mtx(mtx_path)
    print(f"Full graph: |V|={G0.number_of_nodes()} |E|={G0.number_of_edges()} comps={nx.number_connected_components(G0)}")

    G = get_gcc_subgraph(G0) if USE_GCC_ONLY else G0
    print(f"Using {'GCC only' if USE_GCC_ONLY else 'full graph'}: |V|={G.number_of_nodes()} |E|={G.number_of_edges()} comps={nx.number_connected_components(G)}")

    fractions = np.linspace(0.0, MAX_REMOVE, STEPS)

    # Random failures (mean/std)
    random_stats = run_random_trials(
        G,
        fractions,
        trials=RANDOM_TRIALS,
        seed0=0,
        aspl_samples=ASPL_SAMPLES,
    )

    deg_order = removal_order_degree(G)
    deg_res = simulate_progressive_removal(G, deg_order, fractions, aspl_samples=ASPL_SAMPLES, aspl_seed=200)

    core_order = removal_order_coreness(G)
    core_res = simulate_progressive_removal(G, core_order, fractions, aspl_samples=ASPL_SAMPLES, aspl_seed=400)

    title = f"Robustness ({'GCC' if USE_GCC_ONLY else 'Full'}) | max_remove={MAX_REMOVE}, trials={RANDOM_TRIALS}, ASPL samples={ASPL_SAMPLES}"
    plot_gcc_curves(fractions, random_stats, deg_res, core_res=core_res, title=title)
    plot_aspl_curves(fractions, random_stats, deg_res, core_res=core_res, title=title)

    # Save figures 
    plt.savefig("robustness_fast_aspl_lastplot.png", dpi=200, bbox_inches="tight")

    plt.show()