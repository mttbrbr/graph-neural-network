# GNN per Aerodinamica con UniFoil & PyTorch (ROCm per RX 6800)

Questo progetto allena un'architettura **MeshGraphNets** per prevedere il comportamento fluidodinamico attorno a profili alari (airfoils), utilizzando dati reali dal dataset **UniFoil** (composto da 500.000 simulazioni ad alta fedeltà). 

Il codice è stato ripulito dai dati fittizi ed è pronto per l'utilizzo in produzione. È ottimizzato per le schede video AMD tramite driver **ROCm 6.2**.

## Struttura
1. **`setup.sh`**: Script robusto che inizializza da zero l'ambiente, installando `PyTorch` con backend `ROCm 6.2`, `PyTorch Geometric`, tutte le dipendenze matematiche e l'interfaccia GitHub ufficiale `rohitroxkp7/UniFoil`.
2. **`model.py`**: Il cuore della GNN. Codifica i nodi (posizioni, velocità, tipo) e gli archi (distanze) con degli MLP, fa eseguire 6 step di *Message Passing*, ed estrae il nuovo stato fluidodinamico.
3. **`dataset.py`**: Definisce `UniFoilDataset` (classe `Dataset` di PyTorch Geometric). Carica i grafi reali salvati in `.npz` dalla cartella `unifoil_data/raw`.
4. **`extract_real_data.py`**: Script di utilità per parsare i pesanti file di simulazione in formato `.cgns` (scaricati da Harvard) ed estrarre nodi, archi e feature (Velocità `u, v` e pressione `Cp`) tramutandoli nel formato `.npz` compatibile per l'addestramento.
5. **`train.py`**: Loop di addestramento che gira sulla GPU.

## Come iniziare da ZERO

### 1. Preparazione dell'Ambiente
Apri il terminale ed esegui i seguenti comandi:

```bash
# Dai i permessi ed avvia l'installazione robusta
chmod +x setup.sh
./setup.sh

# Attiva l'ambiente
source venv/bin/activate
```

### 2. Ottenere i Dati Reali
Siccome non sono inclusi dati fittizi, devi scaricare i dati reali CFD:

1. Vai su [Harvard Dataverse - UniFoil](https://doi.org/10.7910/DVN/VQGWC4).
2. Scarica i file `.cgns` delle simulazioni desiderate e posizionali dove richiesto dalla libreria UniFoil (generalmente nella stessa cartella da cui esegui lo script).
3. Esegui il convertitore per tramutarli in tensori grafo:
```bash
python extract_real_data.py
```
*Nota: questo estrae l'airfoil 5, caso 2 di esempio. Modifica il file per ciclare su tutto il tuo set di dati scaricato.*

### 3. Addestramento
Una volta che i file `.npz` popolano la cartella `unifoil_data/raw`, lancia l'addestramento:
```bash
python train.py
```
La rete userà la tua **RX 6800** a piena potenza per estrarre la latente di Navier-Stokes 2D!
