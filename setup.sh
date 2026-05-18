#!/bin/bash

# Interrompe lo script in caso di errore
set -e

echo "=== Inizializzazione Ambiente Python ==="
echo "Creazione del virtual environment 'venv'..."
python3 -m venv venv
source venv/bin/activate

echo "Aggiornamento pip..."
pip install --upgrade pip

echo "=== Installazione PyTorch per ROCm 6.2 (AMD RX 6800) ==="
# Installazione specifica per ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

echo "=== Installazione PyTorch Geometric ==="
# Installazione di PyG. Nelle versioni moderne di PyTorch (>=2.0)
# le estensioni c++ sparse non sono strettamente necessarie perché 
# PyTorch integra già le funzioni scatter native ottimizzate.
pip install torch_geometric

echo "=== Installazione librerie accessorie ==="
pip install matplotlib numpy networkx scipy tqdm h5py

echo "=== Installazione interfaccia ufficiale UniFoil ==="
# L'installazione compila e scarica la libreria da GitHub
pip install git+https://github.com/rohitroxkp7/UniFoil.git

echo ""
echo "✅ Setup completato con successo!"
echo "👉 Per attivare l'ambiente esegui: source venv/bin/activate"
