import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from dataset import UniFoilDataset
from model import MeshGraphNet
import os
from tqdm import tqdm

def train():
    # Ottimizzato per AMD ROCm (su Linux `cuda` mappa automaticamente a HIP/ROCm)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware in uso: {device}")
    
    # Inizializziamo il dataset di UniFoil (se non hai i veri file .npz, creerà un mockup)
    dataset = UniFoilDataset(root="unifoil_data")
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    # DataLoader (PyTorch Geometric loottimizza automaticamente per i batch di grafi)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Inizializza il Modello (MeshGraphNet per l'aerodinamica)
    # node_in_dim = 2 (pos: x,y) + 3 (one-hot tipo) + 3 (u, v, p) = 8
    # edge_in_dim = 3 (dx, dy, dist)
    # out_dim = 3 (target u, v, p)
    model = MeshGraphNet(
        node_in_dim=8, 
        edge_in_dim=3, 
        out_dim=3, 
        hidden_dim=128, # Modello più grande per dataset aerodinamici complessi
        num_message_passing_steps=6 
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=1e-4) # Learning rate stabile per GNN
    criterion = nn.MSELoss()
    
    epochs = 30
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch)
            
            # Calcolo Loss (MSE). Generalmente, in ambito fluidodinamico, la loss 
            # si calcola solo sul dominio fluido (evitando l'interno del profilo alare).
            # node_type one-hot: [fluido, airfoil, boundary]. Fluido è indice 0 della one-hot (colonna 2 totale in x).
            fluid_mask = batch.x[:, 2] == 1.0 
            
            loss = criterion(out[fluid_mask], batch.y[fluid_mask])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss = total_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                fluid_mask = batch.x[:, 2] == 1.0
                loss = criterion(out[fluid_mask], batch.y[fluid_mask])
                val_loss += loss.item() * batch.num_graphs
                
        val_loss = val_loss / len(val_dataset)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

if __name__ == "__main__":
    train()
