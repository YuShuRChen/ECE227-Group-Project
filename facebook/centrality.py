import networkx as nx

# read network
G = nx.read_edgelist("facebook_combined.txt", nodetype=int)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# ================================================================================
# part 1： Analyze the nodes centrality in each graph
# Degree centrality
degree_centrality = nx.degree_centrality(G)

# Betweenness centrality
betweenness_centrality = nx.betweenness_centrality(G)

# Eigenvector centrality
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

# find top 5 nodes
def top_k(centrality_dict, k=5):
    return sorted(centrality_dict.items(),
                  key=lambda x: x[1],
                  reverse=True)[:k]

print("\nTop 5 Degree Centrality:")
print(top_k(degree_centrality))

print("\nTop 5 Betweenness Centrality:")
print(top_k(betweenness_centrality))

print("\nTop 5 Eigenvector Centrality:")
print(top_k(eigenvector_centrality))

# ==============================================================================
#part2： How many nodes are among the top 10% in terms of degree centrality
#AND betweenness centrality? How does this overlap change for each of
#these networks?

# number of nodes
n = G.number_of_nodes()
top_k = int(0.1 * n)

print("\nTotal nodes:", n)
print("Top 10% threshold:", top_k)

# degree centrality
degree_sorted = sorted(degree_centrality.items(),
                       key=lambda x: x[1],
                       reverse=True)

# betweenness centrality
betweenness_sorted = sorted(betweenness_centrality.items(),
                            key=lambda x: x[1],
                            reverse=True)

# top 10%
top_degree_nodes = set([node for node, _ in degree_sorted[:top_k]])
top_betweenness_nodes = set([node for node, _ in betweenness_sorted[:top_k]])

# find overlap
overlap = top_degree_nodes & top_betweenness_nodes

print("\nNumber of nodes in top 10% of BOTH degree and betweenness:")
print(len(overlap))

print("\nOverlap percentage relative to top 10%:")
print(len(overlap) / top_k)
