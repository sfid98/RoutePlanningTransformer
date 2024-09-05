import torch
from torch.utils.data import Dataset
import networkx as nx
from random import randint


# Funzione per generare un grafo connesso casuale e calcolare un percorso più breve complesso
def generate_graph_and_shortest_path(num_nodes, num_edges, min_path_length):
    while True:
        G = nx.gnm_random_graph(num_nodes, num_edges)
        if nx.is_connected(G):  # Assicura che il grafo sia connesso
            shortest_path = nx.shortest_path(G, source=0, target=num_nodes - 1)
            if len(shortest_path) >= min_path_length:
                break

    adj_matrix = nx.adjacency_matrix(G).todense()
    return torch.tensor(adj_matrix, dtype=torch.float32), shortest_path

# Dataset personalizzato per gestire il training
class ShortestPathDataset(Dataset):
    def __init__(self, num_graphs, num_nodes, num_edges, min_path_length):
        self.data = []
        self._generate_data(num_graphs, num_nodes, num_edges, min_path_length)

    def _generate_data(self, num_graphs, num_nodes, num_edges, min_path_length):
        for _ in range(num_graphs):
            adj_matrix, shortest_path = generate_graph_and_shortest_path(num_nodes, num_edges, min_path_length)
            max_path_len = num_nodes
            shortest_path_tensor = torch.full((max_path_len,), 0, dtype=torch.long)  # Padding con 0
            if len(shortest_path) > 0:
                shortest_path_tensor[:len(shortest_path)] = torch.tensor(shortest_path, dtype=torch.long)
            self.data.append((adj_matrix, shortest_path_tensor))
            if len(self.data) % 100 == 0:
                print(f'Generated {len(self.data)} graphs')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Parametri del dataset
num_graphs = 1000  # Numero di grafi da generare
num_nodes = 50
num_edges = num_nodes + 10  # Maggiore numero di archi per aumentare la probabilità di connettività
min_path_length = randint(5,10)

 # Lunghezza minima del percorso

# Creazione del dataset
dataset = ShortestPathDataset(num_graphs, num_nodes, num_edges, min_path_length)

# Salvataggio del dataset in un file
torch.save(dataset, 'shortest_path_dataset1.pt')
print("Dataset salvato come 'shortest_path_dataset.pt'")