import torch
from torch.utils.data import Dataset
import networkx as nx
from random import randint

num_nodes = 10

# Definisci i valori per i token SOS e EOS
SOS_TOKEN = num_nodes  # Un valore maggiore rispetto ai nodi del grafo
EOS_TOKEN = num_nodes + 1  # Un valore successivo al SOS_TOKEN

# Funzione per costruire un grafo connesso con un cammino minimo predefinito
def build_graph_with_min_path(num_nodes, num_edges, min_path_length):
    G = nx.path_graph(min_path_length)  # Crea una "spina dorsale" di min_path_length
    additional_edges = num_edges - (min_path_length - 1)

    # Aggiungi i rimanenti nodi e archi casuali per completare il grafo
    for _ in range(num_nodes - min_path_length):
        new_node = len(G.nodes)
        connect_to = randint(0, len(G.nodes) - 1)
        G.add_edge(new_node, connect_to)
    
    while G.number_of_edges() < additional_edges:
        u, v = randint(0, num_nodes - 1), randint(0, num_nodes - 1)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
    
    # Se il grafo non è connesso, lo connetti
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(1, len(components)):
            G.add_edge(list(components[i])[0], list(components[0])[0])

    return G

# Funzione per generare un grafo connesso e calcolare un percorso più breve complesso
def generate_graph_and_shortest_path(num_nodes, num_edges):
    min_path_length = randint(5, num_nodes // 2)
    G = build_graph_with_min_path(num_nodes, num_edges, min_path_length)
    
    adj_matrix = nx.adjacency_matrix(G).todense()

    # Aggiungi righe e colonne per i token SOS e EOS
    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
    sos_row = torch.zeros((1, num_nodes), dtype=torch.float32)
    eos_row = torch.zeros((1, num_nodes), dtype=torch.float32)

    # Estendi la matrice di adiacenza
    adj_matrix = torch.cat((sos_row, adj_matrix, eos_row), dim=0)  # Aggiungi le righe
    sos_col = torch.zeros((num_nodes + 2, 1), dtype=torch.float32)
    eos_col = torch.zeros((num_nodes + 2, 1), dtype=torch.float32)
    adj_matrix = torch.cat((sos_col, adj_matrix, eos_col), dim=1)  # Aggiungi le colonne

    # Genera il percorso più breve con i token SOS e EOS
    shortest_path = [SOS_TOKEN] + nx.shortest_path(G, source=0, target=num_nodes - 1) + [EOS_TOKEN]
    
    return adj_matrix, shortest_path

# Dataset personalizzato per gestire il training
class ShortestPathDataset(Dataset):
    def __init__(self, num_graphs, num_nodes, num_edges):
        self.data = []
        self.num_nodes = num_nodes  # Memorizza num_nodes per l'uso dei token
        self._generate_data(num_graphs, num_nodes, num_edges)

    def _generate_data(self, num_graphs, num_nodes, num_edges):
        for _ in range(num_graphs):
            adj_matrix, shortest_path = generate_graph_and_shortest_path(num_nodes, num_edges)
            max_path_len = num_nodes + 2  # Include spazio per SOS e EOS
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
num_graphs = 500  # Numero di grafi da generare
num_nodes = 10
num_edges = num_nodes + 10  # Numero di archi per aumentare la probabilità di connettività

# Definisci i valori per i token SOS e EOS
SOS_TOKEN = num_nodes  # Un valore maggiore rispetto ai nodi del grafo
EOS_TOKEN = num_nodes + 1  # Un valore successivo al SOS_TOKEN

# Creazione del dataset
dataset = ShortestPathDataset(num_graphs, num_nodes, num_edges)

# Salvataggio del dataset in un file
torch.save(dataset, 'shortest_path_dataset.pt')
print("Dataset salvato come 'shortest_path_dataset.pt'")
