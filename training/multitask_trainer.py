"""
training/multitask_trainer.py
Multi-task training with optional PCGrad.
"""
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from evaluation.metrics import compute_roc_auc
from training.pcgrad import PCGradOptimizer


class MultiTaskTrainer:
    """
    Trainer for multi-task molecular property prediction.
    
    Supports:
    - Standard multi-task training (sum of losses)
    - PCGrad training (gradient conflict mitigation)
    - Per-task evaluation
    
    Parameters
    ----------
    model : nn.Module
        Multi-task model with compute_per_task_losses method
    train_loader : DataLoader
        Multi-task training data
    val_loaders : dict
        Mapping from task_id to validation DataLoader
    cfg : dict
        Training configuration
    device : str
        'cuda' or 'cpu'
    ckpt_path : str
        Checkpoint save path
    use_pcgrad : bool
        Whether to use PCGrad optimizer
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loaders: Dict[int, DataLoader],
        cfg: dict,
        device: str,
        ckpt_path: str,
        use_pcgrad: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loaders = val_loaders
        self.cfg = cfg
        self.device = device
        self.ckpt_path = ckpt_path
        self.use_pcgrad = use_pcgrad
        
        # Optimizer
        base_optimizer = Adam(
            model.parameters(),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("wd", 1e-5),
        )
        
        if use_pcgrad:
            self.optimizer = PCGradOptimizer(base_optimizer, reduction="mean")
        else:
            self.optimizer = base_optimizer
        
        self.scheduler = ReduceLROnPlateau(
            base_optimizer if use_pcgrad else self.optimizer,
            mode="max",
            factor=0.5,
            patience=10,
            verbose=True,
        )
        
        # Tracking
        self.best_avg_auc = 0.0
        self.patience_counter = 0
        self.patience = cfg.get("patience", 30)
        
        self.history = {
            "train_loss": [],
            "val_auc_per_task": [],
            "val_auc_avg": [],
        }
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            
            if self.use_pcgrad:
                # PCGrad training
                self.optimizer.zero_grad()
                
                # Get per-task losses
                task_losses = self.model.compute_per_task_losses(batch)
                
                if len(task_losses) > 0:
                    losses = list(task_losses.values())
                    self.optimizer.backward(losses)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()
                    
                    total_loss += sum(l.item() for l in losses) / len(losses)
                    num_batches += 1
            
            else:
                # Standard training
                self.optimizer.zero_grad()
                loss = self.model.compute_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def eval_task(self, task_id: int, loader: DataLoader) -> float:
        """Evaluate on a single task."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                preds = self.model.predict(batch)
                
                all_preds.append(preds.cpu())
                all_labels.append(batch.y.cpu())
        
        if len(all_preds) == 0:
            return float("nan")
        
        preds = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        return compute_roc_auc(labels, preds)
    
    def eval_all_tasks(self) -> Dict[int, float]:
        """Evaluate on all tasks."""
        results = {}
        for task_id, loader in self.val_loaders.items():
            auc = self.eval_task(task_id, loader)
            results[task_id] = auc
        return results
    
    def run(self, epochs: Optional[int] = None) -> Dict:
        """
        Full training loop.
        
        Returns
        -------
        dict with training history and final results
        """
        epochs = epochs or self.cfg.get("epochs", 200)
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch()
            
            # Evaluate
            task_aucs = self.eval_all_tasks()
            valid_aucs = [v for v in task_aucs.values() if not (v != v)]  # Filter NaN
            avg_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else 0.0
            
            # Learning rate scheduling
            self.scheduler.step(avg_auc)
            
            # Early stopping
            if avg_auc > self.best_avg_auc + 1e-4:
                self.best_avg_auc = avg_auc
                self.patience_counter = 0
                
                # Save checkpoint
                os.makedirs(os.path.dirname(self.ckpt_path) or ".", exist_ok=True)
                torch.save(self.model.state_dict(), self.ckpt_path)
            else:
                self.patience_counter += 1
            
            # Logging
            self.history["train_loss"].append(train_loss)
            self.history["val_auc_per_task"].append(task_aucs)
            self.history["val_auc_avg"].append(avg_auc)
            
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Loss {train_loss:.4f} | "
                f"Val AUC {avg_auc:.4f} | "
                f"Best {self.best_avg_auc:.4f} | "
                f"Patience {self.patience_counter}/{self.patience}"
            )
            
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(
            torch.load(self.ckpt_path, map_location=self.device)
        )
        
        # Final evaluation
        final_aucs = self.eval_all_tasks()
        
        return {
            "history": self.history,
            "best_avg_auc": self.best_avg_auc,
            "final_task_aucs": final_aucs,
        }