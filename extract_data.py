import os
import numpy as np
from scipy.spatial import Delaunay
from unifoil.extract_data import ExtractData

def extract_from_cgns(airfoil_num, case_num, output_dir="unifoil_data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    ed = ExtractData()
    BLOCK = 2 

    try:
        x, y, p = ed.surf_turb(airfoil_num, case_num, "CoefPressure", "extract_xy_quantity", BLOCK)
        _, _, u_x = ed.surf_turb(airfoil_num, case_num, "Velocity", 'b', "extract_xy_quantity", BLOCK)
        _, _, u_y = ed.surf_turb(airfoil_num, case_num, "Velocity", 'c', "extract_xy_quantity", BLOCK)
    except Exception as e:
        print(f"Errore durante l'estrazione per {airfoil_num}-{case_num}: {e}")
        return

    nodes = np.stack([x, y], axis=1)
    dist = np.linalg.norm(nodes, axis=1)
    
    node_type = np.zeros((len(nodes), 3))
    node_type[dist < 1.05, 1] = 1.0
    node_type[dist > 10.0, 2] = 1.0
    node_type[(dist >= 1.05) & (dist <= 10.0), 0] = 1.0

    features = np.stack([u_x, u_y, p], axis=1)
    
    tri = Delaunay(nodes)
    edges = set()
    for s in tri.simplices:
        edges.update([(s[0], s[1]), (s[1], s[2]), (s[2], s[0])])
        
    edge_index = np.array(list(edges)).T
    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)

    src, dst = edge_index
    edge_attr = np.stack([nodes[dst, 0] - nodes[src, 0], nodes[dst, 1] - nodes[src, 1], np.linalg.norm(nodes[dst] - nodes[src], axis=1)], axis=1)

    np.savez_compressed(
        os.path.join(output_dir, f"airfoil_{airfoil_num}_case_{case_num}.npz"),
        nodes=nodes, node_type=node_type, features=features, edge_index=edge_index, edge_attr=edge_attr, target=features
    )
    print(f"Salvato con successo: airfoil_{airfoil_num}_case_{case_num}.npz")

if __name__ == "__main__":
    extract_from_cgns(5, 2)
