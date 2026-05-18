# Graph Neural Network for Aerodynamics

This project implements a **MeshGraphNet** architecture to predict fluid dynamics (velocity and pressure) around airfoils. It trains on real Computational Fluid Dynamics (CFD) data from the **UniFoil** dataset and is fully optimized for GPU training (e.g., AMD via ROCm).

We have extremely simplified the structure to make the workflow clear and straightforward.

## 🗂️ Project Structure

There are now only 3 main files to worry about:
- `setup.sh`: Initializes the environment (installs PyTorch with ROCm, PyTorch Geometric, etc.).
- `extract_data.py`: Reads the heavy CFD simulation files (`.cgns`) and converts them into lightweight Graph Tensors (`.npz`).
- `train.py`: The core file. It contains the Dataset logic, the GNN Model, and the Training loop.

## 🚀 Workflow (How to run it)

### Step 1: Environment Setup
Install all requirements and activate the virtual environment:
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Step 2: Prepare the Data
Since this repository requires real data, you must download the `.cgns` simulation files from the [HuggingFace - UniFoil](https://huggingface.co/datasets/rkanchi/UniFoil/tree/main). 

Place them inside the project folder as required by the unifoil API, then run the extraction script to convert them into Graph formats:
```bash
python extract_data.py
```
*(Note: You can edit `extract_data.py` to loop through multiple airfoils and cases instead of just extracting one)*

### Step 3: Train the Model
Once your data is prepared (the `.npz` files are inside `unifoil_data/raw/`), simply start the training:
```bash
python train.py
```
This script will load the dataset, instantiate the MeshGraphNet model, and start optimizing the network to predict the 2D Navier-Stokes variables (pressure and velocities).
