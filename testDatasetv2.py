import networkx as nx
import osmnx as ox
import random
ox.settings.use_cache = True

# download street network data from OSM and construct a MultiDiGraph model
G = ox.graph.graph_from_point((41.103995, 16.875448), dist=750, network_type="drive")

nodes = list(G.nodes())
random_node1, random_node2 = random.sample(nodes, 2)

# Find the shortest path between the two random nodes
shortest_path = nx.shortest_path(G, source=random_node1, target=random_node2)
print(shortest_path)
# Print the results
print(f"Random Node 1: {random_node1}")
print(f"Random Node 2: {random_node2}")

adj_matrix = nx.adjacency_matrix(G).todense()
print(adj_matrix.shape)