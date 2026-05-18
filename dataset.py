import torch
from torch_geometric.data import Data, Dataset
import numpy as np
import os

class UniFoilDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        Dataset PyTorch Geometric per leggere dati REALI di UniFoil.
        I dati si aspettano sotto forma di file `.npz` estratti dai .cgns originali.
        """
        super().__init__(root, transform, pre_transform)
        self.data_dir = os.path.join(root, "raw")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.file_names = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])
        
        if len(self.file_names) == 0:
            raise FileNotFoundError(
                f"\n[!] NESSUN DATO REALE TROVATO NELLA CARTELLA: {self.data_dir}\n"
                "I dati fittizi sono stati rimossi come richiesto.\n\n"
                "Per procedere con dati reali devi:\n"
                "1. Scaricare le simulazioni UniFoil in formato .cgns da Harvard Dataverse "
                "(https://doi.org/10.7910/DVN/VQGWC4)\n"
                "2. Estrarre i dati dai .cgns in file .npz contenenti i tensori di nodi, archi e feature.\n"
                "   (Puoi usare lo script `extract_real_data.py` che ti ho preparato per automatizzare l'estrazione).\n"
                "3. Posizionare i file .npz generati in 'unifoil_data/raw/'."
            )

    def len(self):
        return len(self.file_names)

    def get(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        data_np = np.load(file_path)
        
        # Carica tensori salvati dallo script di estrazione
        nodes = torch.tensor(data_np['nodes'], dtype=torch.float)
        node_type = torch.tensor(data_np['node_type'], dtype=torch.float)
        features = torch.tensor(data_np['features'], dtype=torch.float)
        edge_index = torch.tensor(data_np['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(data_np['edge_attr'], dtype=torch.float)
        target = torch.tensor(data_np['target'], dtype=torch.float)

        # Concatenazione delle feature di input [x, y, one_hot_type, u, v, p]
        x = torch.cat([nodes, node_type, features], dim=-1)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=target)

if __name__ == "__main__":
    try:
        dataset = UniFoilDataset(root="unifoil_data")
        print(f"Dataset UniFoil REALE caricato. Numero di grafi: {len(dataset)}")
        print(dataset[0])
    except Exception as e:
        print(e)
