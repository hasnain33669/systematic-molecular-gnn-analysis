import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from copy import deepcopy

class Trainer:
    def __init__(self, model, device, task_type, lr=0.001, weight_decay=0.0):
        self.model = model
        self.device = device
        self.task_type = task_type
        
        # Optimizer is configured in run_experiment.py
        self.optimizer = None 
        self.lr = lr
        self.weight_decay = weight_decay

    def set_optimizer(self, optimizer_name):
        if optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer_name == 'NAdam':
            self.optimizer = torch.optim.NAdam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def train(self, dataset):
        self.model.train()
        total_loss = 0
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for data in loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data)
            
            if self.task_type == 'classification':
                loss = F.cross_entropy(out, data.y)
            else: # regression
                loss = F.mse_loss(out.squeeze(), data.y.squeeze())
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * data.num_graphs
            
        return total_loss / len(dataset)

    def evaluate(self, dataset):
        self.model.eval()
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        y_true, y_pred, y_probs = [], [], []
        
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                out = self.model(data)

                if self.task_type == 'classification':
                    y_true.extend(data.y.cpu().numpy())
                    y_pred.extend(out.argmax(dim=1).cpu().numpy())
                    y_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
                else: # regression
                    y_true.extend(data.y.cpu().numpy())
                    y_pred.extend(out.squeeze().cpu().numpy())

        if self.task_type == 'classification':
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "auc_roc": roc_auc_score(y_true, y_probs),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0)
            }
            return metrics
        else: # regression
            mse = mean_squared_error(y_true, y_pred)
            metrics = {
                "mse": mse,
                "rmse": np.sqrt(mse),
                "mae": mean_absolute_error(y_true, y_pred),
                "r2": r2_score(y_true, y_pred)
            }
            return metrics

    def run_training(self, train_dataset, val_dataset, epochs=100):
        best_val_metric = -float('inf') if self.task_type == 'classification' else float('inf')
        best_model_state = None
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train(train_dataset)
            val_metrics = self.evaluate(val_dataset)
            
            # Model selection based on validation metric
            if self.task_type == 'classification':
                current_val_metric = val_metrics['accuracy']
                if current_val_metric > best_val_metric:
                    best_val_metric = current_val_metric
                    best_model_state = deepcopy(self.model.state_dict())
            else: # regression
                current_val_metric = val_metrics['rmse']
                if current_val_metric < best_val_metric:
                    best_val_metric = current_val_metric
                    best_model_state = deepcopy(self.model.state_dict())

            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Metrics: {val_metrics}")

        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        print(f"Training finished. Best validation metric: {best_val_metric:.4f}")
        return self.model

    def evaluate_with_std(self, dataset, n_runs=5, subsample_ratio=0.8):
        """Evaluate with standard deviation using bootstrapping."""
        all_metrics = []
        for _ in range(n_runs):
            indices = np.random.choice(len(dataset), size=int(len(dataset)*subsample_ratio), replace=True)
            subsample = [dataset[i] for i in indices]
            metrics = self.evaluate(subsample)
            all_metrics.append(metrics)

        results_summary = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                results_summary[f"{key}_mean"] = np.mean(values)
                results_summary[f"{key}_std"] = np.std(values)
        return results_summary
