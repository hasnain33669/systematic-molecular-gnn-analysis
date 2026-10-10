# Central configuration for experiments

# Fixed parameters
FIXED_PARAMS = {
    'dim_h': 256,
    'n_layers': 4,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'dropout_rate': 0.10,
    'random_seed': 42,
    'weight_decay': 0.01, 
}

# Varying experimental parameters
EXPERIMENTAL_PARAMS = {
    'models': ['GCN', 'GIN', 'GAT'],
    'optimizers': ['Adam', 'NAdam', 'AdamW'],
    'activations': ['ReLU', 'GELU', 'Swish'],
    'splits': ['random', 'scaffold', 'stratified'],
    'datasets': ['BACE', 'BBBP', 'ESOL', 'FreeSolv'],
}
