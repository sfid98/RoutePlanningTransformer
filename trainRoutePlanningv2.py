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
from datasetBari import ShortestPathDataset

# Impostazioni del modello
n_max = 307 # Numero massimo di nodi (padding target)
num_nodes = n_max  # fix me later
d_model = 512
n_heads = 8
num_layers = 6
dropout = 0.1
d_ff = 2048
batch_size = 10
num_graphs = 50
num_epochs = 10

# Creazione del dataset e del dataloader
dataset = ShortestPathDataset(num_graphs, n_max)

# Divisione del dataset in train e validation set (80% train, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Creazione del modello Transformer
transformer = build_transformer(n_max, d_model, num_layers, n_heads, dropout, d_ff)

# Definizione dell'optimizer e della loss function
optimizer = optim.Adam(transformer.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Funzione per creare la maschera che gestisce il padding
def create_masks(n_nodes, n_max):
    src_mask = torch.zeros((1, n_max, n_max))
    
    # Maschera causale per il decoder che tiene conto del padding
    tgt_mask = torch.triu(torch.ones(n_max, n_max)) == 1
    tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
    
    # Maschera di padding per l'encoder (opzionale, se necessario)
    #padding_mask = torch.arange(n_max).expand(n_nodes, n_max) >= n_nodes
    
    return src_mask, tgt_mask #padding_mask

# Training del modello
transformer.train()
for epoch in range(num_epochs):
    total_loss = 0
    transformer.train()
    for batch in train_loader:
        adj_matrices, shortest_paths = batch
        optimizer.zero_grad()
        
        src_mask, tgt_mask = create_masks(num_nodes, n_max)
        
        encoder_output = transformer.encode(adj_matrices, src_mask)
        decoder_output = transformer.decode(encoder_output, src_mask, shortest_paths, tgt_mask)
        
        logits = transformer.project(decoder_output)
        
        # Calcolo della loss
        loss = criterion(logits.view(-1, n_max), shortest_paths.view(-1))
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
            src_mask, tgt_mask = create_masks(num_nodes, n_max)

            encoder_output = transformer.encode(adj_matrices, src_mask)
            decoder_output = transformer.decode(encoder_output, src_mask, shortest_paths, tgt_mask)
            logits = transformer.project(decoder_output)

            loss = criterion(logits.view(-1, n_max), shortest_paths.view(-1))
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch {epoch}: Train Loss = {avg_train_loss}, Validation Loss = {avg_val_loss}")

# Salvataggio del modello
model_path = 'route_planning_model.pth'
torch.save(transformer.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Testing del modello
transformer.eval()
test_dataset = ShortestPathDataset(10, n_max)  # Generazione di 10 nuovi grafi per il test
test_loader = DataLoader(test_dataset, batch_size=batch_size)

with torch.no_grad():
    for batch in test_loader:
        adj_matrices, shortest_paths = batch
        src_mask, tgt_mask = create_masks(num_nodes, n_max)

        encoder_output = transformer.encode(adj_matrices, src_mask)
        decoder_output = transformer.decode(encoder_output, src_mask, shortest_paths, tgt_mask)
        logits = transformer.project(decoder_output)

        predicted_paths = torch.argmax(logits, dim=-1)
        print("Predicted Paths:", predicted_paths)
        print("Actual Shortest Paths:", shortest_paths)
