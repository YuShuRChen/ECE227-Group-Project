import numpy as np
import networkx as nx
from scipy.io import mmread
import matplotlib.pyplot as plt
import math

MTX_PATH = "ca-Erdos992.mtx"

# -----------------------
# Load graph + GCC
# -----------------------
print("Loading graph...")
A = mmread(MTX_PATH).tocsr()
A.data[:] = 1
G = nx.from_scipy_sparse_array(A)

gcc_nodes = max(nx.connected_components(G), key=len)
Gcc = G.subgraph(gcc_nodes).copy()

n = Gcc.number_of_nodes()
m = Gcc.number_of_edges()
deg = np.array([d for _, d in Gcc.degree()], dtype=int)

print(f"GCC: n={n}, m={m}")
print(f"Average degree = {deg.mean():.4f}")
print(f"Max degree = {deg.max()}")

# -----------------------
# Degree distribution plots
# -----------------------
# Histogram (linear)
plt.figure()
plt.hist(deg, bins=50)
plt.xlabel("Degree k")
plt.ylabel("Count")
plt.title("Degree histogram (linear)")
plt.tight_layout()
plt.savefig("degree_hist_linear.png", dpi=200)
plt.close()

# Log-log plot of P(k) (use counts / n)
vals, counts = np.unique(deg, return_counts=True)
pk = counts / counts.sum()

plt.figure()
plt.loglog(vals, pk, marker="o", linestyle="None")
plt.xlabel("Degree k (log)")
plt.ylabel("P(k) (log)")
plt.title("Degree distribution P(k) (log-log)")
plt.tight_layout()
plt.savefig("degree_dist_loglog.png", dpi=200)
plt.close()

print("Saved plots: degree_hist_linear.png, degree_dist_loglog.png")

# -----------------------
# Poisson sanity check (ER-style baseline)
# -----------------------
lam = deg.mean()
print("\n--- Poisson sanity check ---")
print(f"mean(deg) = {deg.mean():.4f}")
print(f"var(deg)  = {deg.var():.4f}")
print(f"Poisson would expect var ≈ mean (≈ {lam:.4f})")

# -----------------------
# Power-law fitting
# -----------------------
print("\n--- Power-law fit (using powerlaw package) ---")
try:
    import powerlaw
    fit = powerlaw.Fit(deg, discrete=True, verbose=False)
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin
    print(f"Estimated power-law alpha = {alpha:.4f}")
    print(f"Estimated xmin = {xmin}")

    # Compare power-law vs exponential (loglikelihood ratio test)
    R, p = fit.distribution_compare('power_law', 'exponential')
    print(f"Compare power_law vs exponential: R={R:.4f}, p={p:.4g}")
    print("Interpretation: R>0 favors power-law; R<0 favors exponential. Small p means strong evidence.")

    # Compare power-law vs lognormal 
    R2, p2 = fit.distribution_compare('power_law', 'lognormal')
    print(f"Compare power_law vs lognormal:   R={R2:.4f}, p={p2:.4g}")

except Exception as e:
    print("Powerlaw fitting failed:", repr(e))
    print("You can still use the plots + mean/variance check.")

# -----------------------
# Distances: diameter + average shortest path
# -----------------------
print("\n--- Distance metrics (GCC) ---")
try:
    aspl = nx.average_shortest_path_length(Gcc)
    print(f"Average shortest path length (exact) = {aspl:.4f}")
except Exception as e:
    print("Exact ASPL failed/too slow:", repr(e))
    aspl = None

# Exact diameter can be expensive.  Do an approximation using 2-sweep from multiple random sources.
def approx_diameter(G, trials=20, seed=0):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    best = 0
    best_pair = None

    for _ in range(trials):
        s = int(rng.choice(nodes))
        # 1st BFS
        dist = nx.single_source_shortest_path_length(G, s)
        u = max(dist, key=dist.get)
        # 2nd BFS
        dist2 = nx.single_source_shortest_path_length(G, u)
        v = max(dist2, key=dist2.get)
        d = dist2[v]
        if d > best:
            best = d
            best_pair = (u, v)
    return best, best_pair

diam_approx, pair = approx_diameter(Gcc, trials=30, seed=0)
print(f"Approx diameter (2-sweep, 30 trials) = {diam_approx}  (example pair {pair})")

# Optional: approximate ASPL by sampling BFS from random nodes
def approx_aspl(G, samples=200, seed=0):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    total = 0.0
    count = 0
    for s in rng.choice(nodes, size=min(samples, len(nodes)), replace=False):
        dist = nx.single_source_shortest_path_length(G, int(s))
        total += sum(dist.values())
        count += len(dist) - 1
    return total / count

if aspl is None:
    aspl_approx = approx_aspl(Gcc, samples=200, seed=0)
    print(f"Approx average shortest path (200 BFS samples) = {aspl_approx:.4f}")

print("\nDone.")
