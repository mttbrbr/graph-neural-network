import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(out_dim))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x)

class MeshEdgeBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = MLP(node_dim * 2 + edge_dim, hidden_dim, hidden_dim)

    def forward(self, edge_attr, node_features, edge_index):
        src, dst = edge_index
        out = torch.cat([node_features[src], node_features[dst], edge_attr], dim=-1)
        return self.edge_mlp(out)

class MeshNodeBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.node_mlp = MLP(node_dim + edge_dim, hidden_dim, hidden_dim)

    def forward(self, node_features, edge_attr, edge_index):
        src, dst = edge_index
        # Aggregate edge features for each node
        aggr_out = torch.zeros(node_features.size(0), edge_attr.size(1), device=node_features.device)
        aggr_out.scatter_add_(0, dst.unsqueeze(1).expand(-1, edge_attr.size(1)), edge_attr)
        
        out = torch.cat([node_features, aggr_out], dim=-1)
        return self.node_mlp(out)

class MeshGraphNet(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, out_dim, hidden_dim=128, num_message_passing_steps=5):
        super().__init__()
        self.node_encoder = MLP(node_in_dim, hidden_dim, hidden_dim)
        self.edge_encoder = MLP(edge_in_dim, hidden_dim, hidden_dim)
        
        self.processor_steps = num_message_passing_steps
        self.edge_blocks = nn.ModuleList([
            MeshEdgeBlock(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_message_passing_steps)
        ])
        self.node_blocks = nn.ModuleList([
            MeshNodeBlock(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_message_passing_steps)
        ])
        
        self.node_decoder = MLP(hidden_dim, out_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Encoding
        node_features = self.node_encoder(x)
        edge_features = self.edge_encoder(edge_attr)
        
        # Processing (Message Passing)
        for i in range(self.processor_steps):
            edge_features_res = self.edge_blocks[i](edge_features, node_features, edge_index)
            edge_features = edge_features + edge_features_res
            
            node_features_res = self.node_blocks[i](node_features, edge_features, edge_index)
            node_features = node_features + node_features_res
            
        # Decoding
        out = self.node_decoder(node_features)
        
        # Aggiungiamo un'integrazione Euleriana: l'output è la variazione
        # e per le feature fluidodinamiche u, v, p (che sono gli ultimi 3 in input)
        # return data.x[:, -3:] + out # (oppure prediciamo direttamente lo stato futuro)
        
        return out
