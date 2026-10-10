import argparse
import torch

from src.utils import set_seed
from src.data_processing import get_dataset
from src.models import get_model
from src.trainer import Trainer
from experiments.configs import FIXED_PARAMS

def main(args):
    # 1. Setup
    set_seed(FIXED_PARAMS['random_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Load Data
    try:
        train_dataset, val_dataset, test_dataset, task_type = get_dataset(
            dataset_name=args.dataset,
            split_type=args.split,
            device=device
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}")
        return

    # 3. Initialize Model
    model_kwargs = {
        'dim_h': FIXED_PARAMS['dim_h'],
        'n_layers': FIXED_PARAMS['n_layers'],
        'dropout_rate': FIXED_PARAMS['dropout_rate'],
        'activation_name': args.activation,
    }
    model = get_model(args.model, **model_kwargs).to(device)
    
    # Adjust output layer for regression
    if task_type == 'regression':
        model.set_output_dim(1)
    
    print(f"Model: {args.model}, Task: {task_type}, Parameters: {sum(p.numel() for p in model.parameters())}")

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        device=device,
        task_type=task_type,
        lr=FIXED_PARAMS['learning_rate'],
        weight_decay=FIXED_PARAMS['weight_decay']
    )
    trainer.set_optimizer(args.optimizer)

    # 5. Run Training
    print("\nStarting training...")
    trained_model = trainer.run_training(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=FIXED_PARAMS['epochs']
    )

    # 6. Final Evaluation on Test Set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50)
    
    # Single evaluation
    test_metrics = trainer.evaluate(test_dataset)
    print("Single-run test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Evaluation with standard deviation via bootstrapping
    test_results_std = trainer.evaluate_with_std(test_dataset, n_runs=5)
    print("\nBootstrapped test metrics (mean ± std):")
    for key, value in test_results_std.items():
        print(f"  {key}: {value:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a Molecular GNN Benchmark Experiment")
    
    parser.add_argument('--dataset', type=str, required=True, choices=['BACE', 'BBBP', 'ESOL', 'FreeSolv'],
                        help='Name of the dataset to use.')
    parser.add_argument('--model', type=str, required=True, choices=['GCN', 'GIN', 'GAT'],
                        help='Name of the GNN model to use.')
    parser.add_argument('--optimizer', type=str, required=True, choices=['Adam', 'NAdam', 'AdamW'],
                        help='Name of the optimizer to use.')
    parser.add_argument('--activation', type=str, required=True, choices=['ReLU', 'GELU', 'Swish'],
                        help='Name of the activation function to use.')
    parser.add_argument('--split', type=str, required=True, choices=['random', 'scaffold', 'stratified'],
                        help='Type of data splitting strategy to use.')
    
    args = parser.parse_args()
    main(args)
