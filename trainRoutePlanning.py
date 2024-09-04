import networkx as nx
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from modelRoutePlanning import build_transformer

# Funzione per generare un grafo casuale e calcolare il percorso più breve
def generate_graph_and_shortest_path(num_nodes, num_edges):
    G = nx.gnm_random_graph(num_nodes, num_edges)
    adj_matrix = nx.adjacency_matrix(G).todense()
    src, tgt = 0, num_nodes - 1  # Definiamo i nodi di partenza e arrivo
    try:
        shortest_path = nx.shortest_path(G, source=src, target=tgt)
    except nx.NetworkXNoPath:
        shortest_path = []  # Nessun percorso disponibile
    return torch.tensor(adj_matrix, dtype=torch.float32), shortest_path

# Dataset personalizzato per gestire il training
class ShortestPathDataset(Dataset):
    def __init__(self, num_graphs, num_nodes, num_edges):
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.data = []
        for _ in range(num_graphs):
            adj_matrix, shortest_path = generate_graph_and_shortest_path(num_nodes, num_edges)
            max_path_len = num_nodes
            shortest_path_tensor = torch.full((max_path_len,), 0, dtype=torch.long)  # Padding con 0
            if len(shortest_path) > 0:
                shortest_path_tensor[:len(shortest_path)] = torch.tensor(shortest_path, dtype=torch.long)
            self.data.append((adj_matrix, shortest_path_tensor))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Impostazioni del modello
num_nodes = 10
num_edges = 20
d_model = 512
n_heads = 8
num_layers = 6
dropout = 0.1
d_ff = 2048
batch_size = 16
num_graphs = 100

# Creazione del dataset e del dataloader
dataset = ShortestPathDataset(num_graphs, num_nodes, num_edges)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Creazione del modello Transformer
transformer = build_transformer(num_nodes, num_nodes, num_nodes, num_nodes, d_model, num_layers, n_heads, dropout, d_ff)

# Definizione dell'optimizer e della loss function
optimizer = optim.Adam(transformer.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Training del modello
transformer.train()
for epoch in range(10):  # Ridotto il numero di epoch per esempio
    total_loss = 0
    for batch in dataloader:
        adj_matrices, shortest_paths = batch
        optimizer.zero_grad()
        
        src_mask = torch.zeros((1, num_nodes, num_nodes))
        tgt_mask = (torch.triu(torch.ones(num_nodes, num_nodes)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

        encoder_output = transformer.encode(adj_matrices, src_mask)
        decoder_output = transformer.decode(encoder_output, src_mask, shortest_paths, tgt_mask)
        
        logits = transformer.project(decoder_output)
        
        # Calcolo della loss
        loss = criterion(logits.view(-1, num_nodes), shortest_paths.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader)}")

# Testing del modello
transformer.eval()
test_dataset = ShortestPathDataset(10, num_nodes, num_edges)  # Generazione di 10 nuovi grafi per il test
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

with torch.no_grad():
    for batch in test_dataloader:
        adj_matrices, shortest_paths = batch
        src_mask = torch.zeros((1, num_nodes, num_nodes))
        tgt_mask = (torch.triu(torch.ones(num_nodes, num_nodes)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

        encoder_output = transformer.encode(adj_matrices, src_mask)
        decoder_output = transformer.decode(encoder_output, src_mask, shortest_paths, tgt_mask)
        logits = transformer.project(decoder_output)

        predicted_paths = torch.argmax(logits, dim=-1)
        print("Predicted Paths:", predicted_paths)
        print("Actual Shortest Paths:", shortest_paths)
