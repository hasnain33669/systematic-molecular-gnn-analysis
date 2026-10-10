# Molecular GNN Benchmark

This repository contains the code for a systematic benchmark of Graph Neural Networks (GNNs) for molecular property prediction, as described in the manuscript **"Systematic Performance Analysis of Molecular Graph Neural Networks for Property Prediction."**

The benchmark evaluates three GNN architectures—GCN, GIN, and GAT—on four molecular datasets: BACE and BBBP for binary classification, and ESOL and FreeSolv for regression. It systematically examines different data-splitting strategies, optimizers, and activation functions.

## Features

- **Modular Codebase:** Separates data processing, model architectures, training, and experiment execution.
- **Systematic Benchmarking:** Supports configurable experimental parameters for comparative evaluation.
- **Classification and Regression:** Handles binary classification (BACE, BBBP) and regression (ESOL, FreeSolv).
- **Reproducibility:** Uses a fixed random seed to support consistent experimental results.

## Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## File Structure

```text
molecular-gnn-benchmark/
├── .gitignore
├── README.md
├── requirements.txt
├── run_experiment.py
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── models.py
│   ├── trainer.py
│   └── utils.py
├── data/
│   ├── BACE.csv
│   ├── BBBP.csv
│   ├── ESOL.csv
│   └── FreeSolv.csv
└── experiments/
    ├── __init__.py
    └── configs.py
```
