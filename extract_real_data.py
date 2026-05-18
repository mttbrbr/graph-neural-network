import os
import numpy as np
from scipy.spatial import Delaunay
import glob
from tqdm import tqdm

try:
    from unifoil.extract_data import ExtractData
except ImportError:
    print("La libreria unifoil non è installata. Esegui il setup.sh!")
    exit(1)

def extract_from_cgns(airfoil_num, case_num, output_dir="unifoil_data/raw"):
    """
    Usa l'interfaccia UniFoil per leggere un file CGNS (assicurati di aver scaricato i dati 
    nella cartella corretta dove ExtractData se li aspetta) e salvarlo in un grafo .npz.
    """
    os.makedirs(output_dir, exist_ok=True)
    ed = ExtractData()
    
    # Block index = 2 (generalmente usato per il piano z principale della mesh 2D)
    BLOCK = 2 

    print(f"Estrazione dati per Profilo {airfoil_num}, Caso {case_num}...")
    
    try:
        # Estraiamo Pressione (Cp)
        res_cp = ed.surf_turb(
            airfoil_number=airfoil_num,
            case_number=case_num,
            field_name="CoefPressure",
            action="extract_xy_quantity",
            block_index=BLOCK
        )
        if not res_cp:
            print(f"Nessun dato trovato per airfoil {airfoil_num} caso {case_num}. Salto.")
            return

        x, y, p = res_cp

        # Estraiamo Velocità (u_x)
        res_ux = ed.surf_turb(
            airfoil_number=airfoil_num,
            case_number=case_num,
            field_name="Velocity",
            vel_component='b', # 'b' = u_x
            action="extract_xy_quantity",
            block_index=BLOCK
        )
        _, _, u_x = res_ux
        
        # Estraiamo Velocità (u_y)
        res_uy = ed.surf_turb(
            airfoil_number=airfoil_num,
            case_number=case_num,
            field_name="Velocity",
            vel_component='c', # 'c' = u_y
            action="extract_xy_quantity",
            block_index=BLOCK
        )
        _, _, u_y = res_uy

        # 1. Creiamo l'array dei nodi
        nodes = np.stack([x, y], axis=1)
        num_nodes = len(nodes)

        # 2. Assegniamo le tipologie (one-hot: [fluido, airfoil, boundary])
        # Per semplicità qui calcoliamo bounding box / distanza dal centro per separare i tipi
        # Nei dati reali CFD, idealmente si estraggono i flag di boundary condition dal CGNS
        node_type = np.zeros((num_nodes, 3))
        dist_from_origin = np.sqrt(x**2 + y**2)
        
        # Semplice euristica per distinguere i nodi (aggiustare con i veri marker del mesh CGNS se disponibili)
        is_airfoil = dist_from_origin < 1.05 # L'airfoil generalmente sta vicino all'origine [0,1]x[-0.5,0.5]
        is_boundary = dist_from_origin > 10.0 # Outer boundary molto lontana
        
        node_type[is_airfoil, 1] = 1.0
        node_type[is_boundary, 2] = 1.0
        node_type[~(is_airfoil | is_boundary), 0] = 1.0 # Resto è fluido

        # 3. Features correnti: [u_x, u_y, p]
        features = np.stack([u_x, u_y, p], axis=1)
        
        # (Opzionale) Target: essendo simulazioni stazionarie, o estrai il passo temporale successivo
        # oppure in un setup stazionario (steady-state) cerchi di predire p e u_x, u_y dalla sola geometria.
        # Qui usiamo un target fittizio che è lo stato stesso per fare auto-encoding stazionario.
        target = np.copy(features)

        # 4. Creiamo gli archi tramite Triangolazione di Delaunay (Mesh connectivity proxy)
        print("Calcolo connettività mesh (Delaunay)...")
        tri = Delaunay(nodes)
        
        # Estraiamo gli archi dai triangoli
        edges = set()
        for simplex in tri.simplices:
            edges.add((simplex[0], simplex[1]))
            edges.add((simplex[1], simplex[2]))
            edges.add((simplex[2], simplex[0]))
            
        edge_index = np.array(list(edges)).T
        
        # Rendiamo bidirezionale
        edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)

        # 5. Attributi archi
        src, dst = edge_index
        dx = nodes[dst, 0] - nodes[src, 0]
        dy = nodes[dst, 1] - nodes[src, 1]
        dist = np.sqrt(dx**2 + dy**2)
        edge_attr = np.stack([dx, dy, dist], axis=1)

        # 6. Salvataggio in NPZ
        out_file = os.path.join(output_dir, f"airfoil_{airfoil_num}_case_{case_num}.npz")
        np.savez_compressed(
            out_file,
            nodes=nodes,
            node_type=node_type,
            features=features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            target=target
        )
        print(f"Salvato con successo: {out_file}")

    except Exception as e:
        print(f"Errore durante l'estrazione: {e}")

if __name__ == "__main__":
    print("=== Tool di Estrazione UniFoil CGNS -> NPZ ===")
    print("Assicurati di aver scaricato i file .cgns nella struttura cartelle richiesta dalla libreria unifoil.")
    print("Sto tentando di estrarre il Profilo 5, Caso 2 come esempio...\n")
    
    extract_from_cgns(airfoil_num=5, case_num=2)
    
    print("\nModifica questo script inserendo un loop per processare tutti gli airfoil e casi che hai scaricato.")
