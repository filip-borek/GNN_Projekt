# MPNN for LogP Prediction

Graph Neural Network experiments for predicting the octanol–water partition coefficient (LogP) of organic molecules from SMILES strings.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/filip-borek/GNN_Projekt/blob/main/01_Eksperymenty.ipynb)

## Overview

This project implements Message Passing Neural Networks (MPNN) built with [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/). Molecules are converted from SMILES into graph representations (atoms as nodes, bonds as edges), and several model variants are trained and compared on a LogP regression task.

## Experiments

| Model | Layers | Global pooling |
|-------|--------|----------------|
| `MPNN` | 3 | Mean |
| `MPNNAdd` | 3 | Sum |
| `MPNNMultiPool` | 3 | Mean + Max + Sum |
| `MPNNAdd5Layers` | 5 | Sum |

Models are evaluated using validation **MSE** and **MAE**.

## Dataset

Molecules are fetched at runtime from the [xlogp dataset](https://github.com/dgront/chem-ml) (JChemEdu). SMILES strings are parsed with RDKit; invalid structures are skipped automatically.

## Project structure

```
GNN_Projekt/
├── 01_Eksperymenty.ipynb   # Main notebook (experiments & plots)
├── data_processing.py      # SMILES → PyG graphs, train/val/test split
├── trainer.py              # Training loop
└── models/
    └── mpnn.py             # MPNN architectures
```

## Run in Google Colab

1. Open the notebook via the **Open in Colab** badge above.
2. Select **Runtime → Run all**.
3. The first code cell clones this repository and installs dependencies.

> The repository must be **public** so Colab can clone it without authentication.

## Dependencies

Installed automatically in Colab:

- PyTorch / PyTorch Geometric
- RDKit (`rdkit-pypi`)
- matplotlib, tqdm, requests

## License

Academic project — see course requirements for usage terms.
