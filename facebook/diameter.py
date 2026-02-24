import networkx as nx

G = nx.read_edgelist("facebook_combined.txt", nodetype=int)

if not nx.is_connected(G):
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

diameter = nx.diameter(G)
avg_path = nx.average_shortest_path_length(G)

print("Diameter:", diameter)
print("Average shortest path length:", avg_path)