import networkx as nx
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
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

# Funzione di greedy decoding
def greedy_decode(transformer, encoder_output, src_mask, max_len, sos_token, eos_token):
    ys = torch.ones(1, 1).fill_(sos_token).type(torch.long).to(encoder_output.device)
    for _ in range(max_len - 1):
        tgt_mask = (torch.triu(torch.ones(ys.size(1), ys.size(1))) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

        out = transformer.decode(encoder_output, src_mask, ys, tgt_mask)
        prob = transformer.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        
        next_word_item = next_word.item()

        ys = torch.cat([ys, next_word.unsqueeze(0)], dim=1)

        if next_word_item == eos_token:
            break
    return ys

# Definizione dei parametri
num_nodes = 10
d_model = 512
n_heads = 8
num_layers = 6
dropout = 0.1
d_ff = 2048
batch_size = 16
num_epochs = 10
max_len = num_nodes + 2

# Caricamento del dataset
loaded_data = torch.load('shortest_path_dataset.pt')

dataset = ShortestPathDataset(data=loaded_data)

print(f"Dataset caricato con {len(dataset)} grafi.")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Creazione del modello Transformer
transformer = build_transformer(num_nodes + 2, num_nodes + 2, num_nodes + 2, num_nodes + 2, d_model, num_layers, n_heads, dropout, d_ff)

optimizer = optim.Adam(transformer.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Training del modello
transformer.train()
for epoch in range(num_epochs):
    total_loss = 0
    transformer.train()
    for batch in train_loader:
        adj_matrices, shortest_paths = batch
        optimizer.zero_grad()

        # Correzione delle dimensioni delle maschere
        src_mask = torch.zeros((adj_matrices.size(0), 1, 1, num_nodes + 2)).to(adj_matrices.device)
        
        tgt_mask = (torch.triu(torch.ones(num_nodes + 2, num_nodes + 2)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

        encoder_output = transformer.encode(adj_matrices, src_mask)
        decoder_output = transformer.decode(encoder_output, src_mask, shortest_paths, tgt_mask)

        logits = transformer.project(decoder_output)

        # Calcolo della loss
        loss = criterion(logits.view(-1, num_nodes + 2), shortest_paths.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}: Train Loss = {avg_train_loss}")

    # Validation con greedy decoding
    transformer.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            adj_matrices, shortest_paths = batch
            print(adj_matrices)
            src_mask = torch.zeros((1, 1, 1, num_nodes + 2)).to(adj_matrices.device)

            encoder_output = transformer.encode(adj_matrices, src_mask)

            sos_token = num_nodes
            eos_token = num_nodes + 1

            # Predizione del percorso con greedy decoding
            predicted_paths = greedy_decode(transformer, encoder_output, src_mask, max_len, sos_token, eos_token)

            # Calcolo della loss tra il percorso predetto e quello reale
            predicted_paths_padded = F.pad(predicted_paths, (0, max_len - predicted_paths.size(1)), value=eos_token)

            # Converti predicted_paths_padded al tipo Float
            predicted_paths_padded = predicted_paths_padded.float()

            shortest_paths = shortest_paths.float()
            print(f"predicted_paths_padded shape: {predicted_paths_padded.shape}")
            print(f"shortest_paths shape: {shortest_paths.shape}")
            loss = criterion(predicted_paths_padded.view(-1), shortest_paths.view(-1))
            val_loss += loss.item()

            print(f"Predicted Paths: {predicted_paths}")
            print(f"Actual Shortest Paths: {shortest_paths}")

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss = {avg_val_loss}")

# Testing del modello
transformer.eval()

loaded_test_data = torch.load('shortest_path_dataset1.pt')

test_dataset = ShortestPathDataset(data=loaded_test_data)
print("Dataset caricato da 'shortest_path_dataset1.pt'")
test_loader = DataLoader(test_dataset, batch_size=batch_size)

with torch.no_grad():
    for batch in test_loader:
        adj_matrices, shortest_paths = batch
        src_mask = torch.zeros((adj_matrices.size(0), 1, 1, num_nodes + 2)).to(adj_matrices.device)

        encoder_output = transformer.encode(adj_matrices, src_mask)

        sos_token = num_nodes
        eos_token = num_nodes + 1

        # Predizione del percorso con greedy decoding
        predicted_paths = greedy_decode(transformer, encoder_output, src_mask, max_len, sos_token, eos_token)

        print("Predicted Paths:", predicted_paths)
        print("Actual Shortest Paths:", shortest_paths)