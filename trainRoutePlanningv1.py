import networkx as nx
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from sklearn.model_selection import train_test_split
from modelRoutePlanningv1 import build_transformer
from random import randint


# Dataset personalizzato per gestire il training (come sopra)
class ShortestPathDataset(Dataset):
    def __init__(self, data=None, num_graphs=None, num_nodes=None, num_edges=None):
        if data:
            self.data = data
        else:
            self.data = []
            self._generate_data(num_graphs, num_nodes, num_edges)

    def _generate_data(self, num_graphs, num_nodes, num_edges):
        # Funzione per generare i dati (come sopra)
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


num_nodes = 50
d_model = 512
n_heads = 8
num_layers = 6
dropout = 0.1
d_ff = 2048
batch_size = 16
num_epochs = 10
# Caricamento del dataset
loaded_data = torch.load('shortest_path_dataset.pt')

# Creazione di un'istanza di ShortestPathDataset usando i dati caricati
dataset = ShortestPathDataset(data=loaded_data)

print(f"Dataset caricato con {len(dataset)} grafi.")

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

loaded_test_data = torch.load('shortest_path_dataset1.pt')

# Creazione di un'istanza di ShortestPathDataset usando i dati caricati
test_dataset = ShortestPathDataset(data=loaded_test_data)
print("Dataset caricato da 'shortest_path_dataset1.pt'")
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