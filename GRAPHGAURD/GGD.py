import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt

# 1. Δημιουργία graph
edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 5, 1, 2],
    [1, 0, 3, 2, 5, 4, 4, 5]
], dtype=torch.long)

# Features: Οι κόμβοι 2 και 4 έχουν "ύποπτα" υψηλά χαρακτηριστικά
x = torch.tensor([[1.0], [1.0], [10.0], [1.0], [10.0], [1.0]], dtype=torch.float)

# Labels: 0 = Normal, 1 = Malicious
y = torch.tensor([0, 0, 1, 0, 1, 0], dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

# 2. Ορισμός GCN (Πιο απλό για να αποφύγουμε το over-smoothing)
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(1, 2) 
        self.conv2 = GCNConv(2, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN()

# Αυξάνουμε το Learning Rate (lr) για να μαθαίνει πιο επιθετικά
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# Δίνουμε ΒΑΡΟΣ στην κλάση 1 (Malicious) για να αναγκάσουμε το AI να την προσέξει
# Το [1.0, 3.0] σημαίνει: "Τιμώρησε το λάθος στους κακούς 3 φορές περισσότερο"
class_weights = torch.tensor([1.0, 3.0])

# 3. Training loop
print("Starting aggressive training...")
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    # Εφαρμογή των weights στο loss function
    loss = F.nll_loss(out, data.y, weight=class_weights)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

print("Training complete!")

# 4. Predictions
pred = out.argmax(dim=1)
print("\nReal Labels: ", y.tolist())
print("Predictions: ", pred.tolist())

# 5. Visualization
G = nx.Graph()
G.add_edges_from(edge_index.t().tolist())

color_map = []
for node_idx in range(len(pred)):
    if pred[node_idx] == 1:
        color_map.append("red")   # Identified as Malicious
    else:
        color_map.append("green") # Identified as Normal

plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color=color_map, node_size=900, font_weight='bold')
plt.title("GGD: AI-Based Intrusion Detection Results")
plt.show()