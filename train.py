import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# ==========================================
# 1. DATASET
# ==========================================
class UniFoilDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data_dir = os.path.join(root, "raw")
        os.makedirs(self.data_dir, exist_ok=True)
        self.file_names = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])

        if not self.file_names:
            print(f"[!] No data found in {self.data_dir}. Please run 'extract_data.py' first.")

    def len(self):
        return len(self.file_names)

    def get(self, idx):
        data = np.load(os.path.join(self.data_dir, self.file_names[idx]))
        
        nodes = torch.tensor(data['nodes'], dtype=torch.float)
        node_type = torch.tensor(data['node_type'], dtype=torch.float)
        features = torch.tensor(data['features'], dtype=torch.float)
        
        # Combine [x, y, one_hot_type, u, v, p]
        x = torch.cat([nodes, node_type, features], dim=-1)
        
        return Data(
            x=x,
            edge_index=torch.tensor(data['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(data['edge_attr'], dtype=torch.float),
            y=torch.tensor(data['target'], dtype=torch.float)
        )

# ==========================================
# 2. MODEL (MeshGraphNet)
# ==========================================
class MLP(nn.Sequential):
    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super().__init__(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        )

class MeshGraphNet(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, out_dim, hidden_dim=128, steps=5):
        super().__init__()
        self.steps = steps
        self.node_enc = MLP(node_in_dim, hidden_dim)
        self.edge_enc = MLP(edge_in_dim, hidden_dim)
        
        self.edge_mlps = nn.ModuleList([MLP(hidden_dim * 3, hidden_dim) for _ in range(steps)])
        self.node_mlps = nn.ModuleList([MLP(hidden_dim * 2, hidden_dim) for _ in range(steps)])
        
        self.dec = MLP(hidden_dim, out_dim)

    def forward(self, data):
        node_feats = self.node_enc(data.x)
        edge_feats = self.edge_enc(data.edge_attr)
        src, dst = data.edge_index
        
        # Message Passing
        for i in range(self.steps):
            # Update edges
            edge_feats = edge_feats + self.edge_mlps[i](torch.cat([node_feats[src], node_feats[dst], edge_feats], dim=-1))
            
            # Update nodes
            aggr = torch.zeros(node_feats.size(0), edge_feats.size(1), device=node_feats.device)
            aggr.scatter_add_(0, dst.unsqueeze(1).expand(-1, edge_feats.size(1)), edge_feats)
            node_feats = node_feats + self.node_mlps[i](torch.cat([node_feats, aggr], dim=-1))
            
        return self.dec(node_feats)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = UniFoilDataset(root="unifoil_data")
    if len(dataset) == 0:
        return

    # Train/Val split
    train_size = int(0.8 * len(dataset))
    if train_size == 0 and len(dataset) > 0:
        train_size = len(dataset)
        
    train_loader = DataLoader(dataset[:train_size], batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset[train_size:], batch_size=4, shuffle=False)
    
    # Model parameters: 8 (node features), 3 (edge features), 3 (output targets)
    model = MeshGraphNet(node_in_dim=8, edge_in_dim=3, out_dim=3, hidden_dim=128, steps=6).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(30):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            
            # Mask to calculate loss ONLY on the fluid domain (idx 2 in node type)
            fluid_mask = batch.x[:, 2] == 1.0 
            loss = criterion(out[fluid_mask], batch.y[fluid_mask])
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                fluid_mask = batch.x[:, 2] == 1.0
                val_loss += criterion(model(batch)[fluid_mask], batch.y[fluid_mask]).item() * batch.num_graphs
                
        train_l = train_loss / max(1, len(dataset[:train_size]))
        val_l = val_loss / max(1, len(dataset[train_size:]))
        
        print(f"Epoch {epoch+1} | Train Loss: {train_l:.6f} | Val Loss: {val_l:.6f}")

if __name__ == "__main__":
    train()
