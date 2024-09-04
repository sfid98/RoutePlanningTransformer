import networkx as nx
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from sklearn.model_selection import train_test_split
from modelRoutePlanning import build_transformer
import osmnx as ox
import random
import itertools


# Carica il grafo stradale
def load_graph():
    G = ox.graph.graph_from_point((41.103995, 16.875448), dist=750, network_type="drive")
    return G


# Funzione per creare una mappatura dei nodi
def create_node_mapping(G):
    node_mapping = {node: idx for idx, node in enumerate(G.nodes)}
    return node_mapping

# Funzione per trasformare il grafo con nodi rinumerati
def renumber_nodes(G, node_mapping):
    G_renumbered = nx.relabel_nodes(G, node_mapping)
    return G_renumbered

def generate_graph_data(G, node_mapping, src, tgt, n_max):
    G_renumbered = renumber_nodes(G, node_mapping)
    
    # Calcola il percorso più breve
    try:
        shortest_path = nx.shortest_path(G_renumbered, source=src, target=tgt, weight='length')
    except nx.NetworkXNoPath:
        return None, None
    
    # Crea la matrice di adiacenza e applica il padding
    adj_matrix = nx.adjacency_matrix(G_renumbered).todense()
    num_nodes = len(G_renumbered.nodes)
    adj_matrix_padded = torch.zeros((n_max, n_max), dtype=torch.float32)
    adj_matrix_padded[:num_nodes, :num_nodes] = torch.tensor(adj_matrix, dtype=torch.float32)
    
    # Padding del percorso più breve
    max_path_len = n_max
    shortest_path_tensor = torch.full((max_path_len,), 0, dtype=torch.long)  # Padding con -1
    if len(shortest_path) > 0:
        shortest_path_tensor[:len(shortest_path)] = torch.tensor(shortest_path, dtype=torch.long)    
    return adj_matrix_padded, shortest_path_tensor


# Dataset personalizzato per gestire il training
class ShortestPathDataset(Dataset):
    def __init__(self, num_graphs, n_max):
        self.num_graphs = num_graphs
        self.n_max = n_max
        self.G = load_graph()
        self.node_mapping = create_node_mapping(self.G)
        self.G_renumbered = renumber_nodes(self.G, self.node_mapping)
        
        # Genera tutte le coppie di nodi
        nodes = list(self.G_renumbered.nodes())
        print("numero permutazioni")
        print(len(list(itertools.permutations(nodes, 2))))
        self.pairs = [(src, tgt) for src, tgt in itertools.permutations(nodes, 2)]
        
        # Seleziona un sottoinsieme di coppie
        self.pairs = random.sample(self.pairs, num_graphs)
        
        self.data = []
        for src, tgt in self.pairs:
            adj_matrix, shortest_path = generate_graph_data(self.G, self.node_mapping, src, tgt, n_max)
            if adj_matrix is not None and shortest_path is not None:
                self.data.append((adj_matrix, shortest_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]