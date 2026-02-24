"""
training/trainer.py
Central training harness: early stopping, ROC-AUC evaluation, checkpoint saving.
"""
import os
import random
from typing import Optional

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluation.metrics import compute_roc_auc


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Fix all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Tracks validation ROC-AUC and triggers early stopping.
    Saves the best model checkpoint automatically.

    Parameters
    ----------
    patience  : int   – epochs to wait without improvement before stopping
    min_delta : float – minimum improvement to reset the patience counter
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_score: float | None = None
        self.stop      = False

    def step(self, score: float, model: torch.nn.Module, ckpt_path: str):
        """
        Call once per epoch with the current validation score.
        Saves checkpoint when score improves; sets self.stop when patience runs out.
        """
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Generic single-task trainer.

    The model must expose:
      - model.compute_loss(batch) -> scalar tensor
      - model.predict(batch)      -> sigmoid probabilities  [N, num_tasks]

    Parameters
    ----------
    model        : nn.Module
    train_loader : DataLoader
    val_loader   : DataLoader
    cfg          : dict  – hyperparameter dict (lr, wd, patience, epochs)
    device       : str   – 'cuda' or 'cpu'
    ckpt_path    : str   – path where best checkpoint is saved
    """

    def __init__(self, model, train_loader, val_loader, cfg, device, ckpt_path):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device
        self.ckpt_path    = ckpt_path

        self.optimizer = Adam(
            model.parameters(),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("wd", 1e-5),
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=10,
            verbose=True,
        )
        self.es = EarlyStopping(patience=cfg.get("patience", 30))

        self.history = {"train_loss": [], "val_auc": []}

    # ── single epoch ─────────────────────────────────────────────────────────

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(self.train_loader), 1)

    def eval_epoch(self, loader) -> float:
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                preds = self.model.predict(batch)
                all_preds.append(preds.cpu())
                all_labels.append(batch.y.cpu())
        return compute_roc_auc(
            torch.cat(all_labels, dim=0),
            torch.cat(all_preds,  dim=0),
        )

    # ── full training run ────────────────────────────────────────────────────

    def run(self, epochs: Optional[int] = None) -> float:
        """
        Train for up to `epochs` epochs (default: cfg['epochs']).
        Returns the best validation ROC-AUC seen.
        """
        epochs = epochs or self.cfg.get("epochs", 200)
        for ep in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_auc    = self.eval_epoch(self.val_loader)

            self.scheduler.step(val_auc)
            self.es.step(val_auc, self.model, self.ckpt_path)

            self.history["train_loss"].append(train_loss)
            self.history["val_auc"].append(val_auc)

            print(
                f"Epoch {ep:3d}/{epochs} | "
                f"Loss {train_loss:.4f} | "
                f"Val AUC {val_auc:.4f} | "
                f"Best {self.es.best_score:.4f} | "
                f"Patience {self.es.counter}/{self.es.patience}"
            )

            if self.es.stop:
                print(f"Early stopping triggered at epoch {ep}.")
                break

        # Restore best weights
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        return self.es.best_score