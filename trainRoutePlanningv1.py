import networkx as nx
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from sklearn.model_selection import train_test_split
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
num_epochs = 10

# Creazione del dataset e del dataloader
dataset = ShortestPathDataset(num_graphs, num_nodes, num_edges)

# Divisione del dataset in train e validation set (80% train, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Creazione del modello Transformer
transformer = build_transformer(num_nodes, num_nodes, num_nodes, num_nodes, d_model, num_layers, n_heads, dropout, d_ff)

# Definizione dell'optimizer e della loss function
optimizer = optim.Adam(transformer.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Training del modello
transformer.train()
for epoch in range(num_epochs):  # Ridotto il numero di epoch per esempio
    total_loss = 0
    transformer.train()
    for batch in train_loader:
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
    
    avg_train_loss = total_loss / len(train_loader)
    
    # Validation
    transformer.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            adj_matrices, shortest_paths = batch
            src_mask = torch.zeros((1, num_nodes, num_nodes))
            tgt_mask = (torch.triu(torch.ones(num_nodes, num_nodes)) == 1).transpose(0, 1)
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

            encoder_output = transformer.encode(adj_matrices, src_mask)
            decoder_output = transformer.decode(encoder_output, src_mask, shortest_paths, tgt_mask)
            logits = transformer.project(decoder_output)

            loss = criterion(logits.view(-1, num_nodes), shortest_paths.view(-1))
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch {epoch}: Train Loss = {avg_train_loss}, Validation Loss = {avg_val_loss}")

# Salvataggio del modello
model_path = 'route_planning_model.pth'
torch.save(transformer.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Testing del modello
transformer.eval()
test_dataset = ShortestPathDataset(10, num_nodes, num_edges)  # Generazione di 10 nuovi grafi per il test
test_loader = DataLoader(test_dataset, batch_size=batch_size)

with torch.no_grad():
    for batch in test_loader:
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
