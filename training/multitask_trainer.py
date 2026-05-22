"""
training/multitask_trainer.py
Multi-task trainer with PCGrad support and device-safe execution.

Fixes:
    1. Explicit device placement for all batch transfers
    2. PCGrad stats logging per epoch
    3. Resume-from-checkpoint support
    4. Per-epoch conflict statistics saved for paper analysis
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np
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

    Parameters
    ----------
    model       : nn.Module with compute_per_task_losses() and predict()
    train_loader : DataLoader — multi-task batches
    val_loaders  : dict[task_id -> DataLoader]
    cfg          : dict — lr, wd, patience, epochs
    device       : str — 'cuda' or 'cpu'
    ckpt_path    : str — checkpoint save path
    use_pcgrad   : bool — enable PCGrad
    log_pcgrad_stats : bool — save per-epoch conflict stats
    resume       : bool — resume from checkpoint if exists
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
        log_pcgrad_stats: bool = True,
        resume: bool = False,
    ):
        # ── Device setup ─────────────────────────────────────────────────
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loaders = val_loaders
        self.cfg = cfg
        self.ckpt_path = ckpt_path
        self.use_pcgrad = use_pcgrad
        self.log_pcgrad_stats = log_pcgrad_stats

        # ── Optimizer ────────────────────────────────────────────────────
        base_optimizer = Adam(
            model.parameters(),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("wd", 1e-5),
        )

        if use_pcgrad:
            self.optimizer = PCGradOptimizer(
                base_optimizer,
                reduction="mean",
                max_norm=1.0,
                log_stats=log_pcgrad_stats,
            )
            self._base_optimizer = base_optimizer
        else:
            self.optimizer = base_optimizer
            self._base_optimizer = base_optimizer

        # ── LR Scheduler ─────────────────────────────────────────────────
        self.scheduler = ReduceLROnPlateau(
            self._base_optimizer,
            mode="max",
            factor=0.5,
            patience=10,
        )

        # ── Tracking ─────────────────────────────────────────────────────
        self.best_avg_auc = 0.0
        self.patience_counter = 0
        self.patience = cfg.get("patience", 30)
        self.start_epoch = 1
        self._use_cuda_amp = torch.cuda.is_available() and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._use_cuda_amp)

        self.history = {
            "train_loss": [],
            "val_auc_per_task": [],
            "val_auc_avg": [],
            "pcgrad_stats": [],  # Per-epoch conflict stats
        }

        # ── Resume ───────────────────────────────────────────────────────
        if resume and os.path.exists(ckpt_path):
            self._resume_from_checkpoint()

    def _resume_from_checkpoint(self):
        """Load checkpoint and restore training state."""
        resume_path = self.ckpt_path.replace(".pt", "_resume.pt")
        load_path = resume_path if os.path.exists(resume_path) else self.ckpt_path
        print(f"  Resuming from: {load_path}")

        state = torch.load(load_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(state["model"])
        self._base_optimizer.load_state_dict(state["optimizer"])
        self.best_avg_auc = state.get("best_avg_auc", 0.0)
        self.patience_counter = state.get("patience_counter", 0)
        self.start_epoch = state.get("epoch", 0) + 1
        self.history = state.get("history", self.history)

        print(f"  Resumed at epoch {self.start_epoch - 1}, best AUC {self.best_avg_auc:.4f}")

    def _save_checkpoint(self, epoch: int):
        """Save full training state for resume support."""
        os.makedirs(os.path.dirname(self.ckpt_path) or ".", exist_ok=True)

        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self._base_optimizer.state_dict(),
            "best_avg_auc": self.best_avg_auc,
            "patience_counter": self.patience_counter,
            "epoch": epoch,
            "history": self.history,
        }, self.ckpt_path)

    def _move_batch(self, batch):
        """Move batch to device safely."""
        return batch.to(self.device)

    # ── Training Epoch ────────────────────────────────────────────────────

    def train_epoch(self) -> tuple:
        """
        Train for one epoch.

        Returns
        -------
        avg_loss : float
        pcgrad_stats : dict (empty if not using PCGrad)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_pcgrad_stats = []

        for batch in self.train_loader:
            # ── Device transfer ──────────────────────────────────────
            batch = self._move_batch(batch)

            if self.use_pcgrad:
                # ── PCGrad path ──────────────────────────────────────
                with torch.amp.autocast("cuda", enabled=self._use_cuda_amp):
                    task_losses = self.model.compute_per_task_losses(batch)

                if not task_losses:
                    continue

                losses = list(task_losses.values())

                # Verify all losses are on correct device
                for loss in losses:
                    assert loss.device.type == self.device.type, (
                        f"Loss is on {loss.device}, model is on {self.device}. "
                        "Check that all model operations stay on GPU."
                    )

                self.optimizer.zero_grad()
                scaled_losses = [self.scaler.scale(loss) for loss in losses]
                orig_max_norm = self.optimizer.max_norm
                if self._use_cuda_amp:
                    self.optimizer.max_norm = 0.0
                stats = self.optimizer.backward(scaled_losses)
                if self._use_cuda_amp:
                    self.optimizer.max_norm = orig_max_norm
                self.scaler.unscale_(self._base_optimizer)
                if self._use_cuda_amp and orig_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=orig_max_norm
                    )
                self.scaler.step(self._base_optimizer)
                self.scaler.update()

                if stats and self.log_pcgrad_stats:
                    epoch_pcgrad_stats.append(stats)

                batch_loss = sum(l.item() for l in losses) / len(losses)

            else:
                # ── Standard path ────────────────────────────────────
                self.optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=self._use_cuda_amp):
                    loss = self.model.compute_loss(batch)

                assert loss.device.type == self.device.type, (
                    f"Loss device mismatch: {loss.device} vs {self.device}"
                )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self._base_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                self.scaler.step(self._base_optimizer)
                self.scaler.update()
                batch_loss = loss.item()

            total_loss += batch_loss
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Aggregate PCGrad stats for this epoch
        epoch_stats = {}
        if epoch_pcgrad_stats:
            keys = epoch_pcgrad_stats[0].keys()
            epoch_stats = {
                k: float(np.mean([s[k] for s in epoch_pcgrad_stats]))
                for k in keys
            }

        return avg_loss, epoch_stats

    # ── Evaluation ────────────────────────────────────────────────────────

    def eval_task(self, task_id: int, loader: DataLoader) -> float:
        """Evaluate model on a single task."""
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                batch = self._move_batch(batch)
                preds = self.model.predict(batch)

                # Ensure preds come back to CPU for metric computation
                all_preds.append(preds.cpu())
                all_labels.append(batch.y.cpu())

        if not all_preds:
            return float("nan")

        preds = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)

        return compute_roc_auc(labels, preds)

    def eval_all_tasks(self) -> Dict[int, float]:
        """Evaluate all tasks, return dict of task_id -> AUC."""
        return {
            task_id: self.eval_task(task_id, loader)
            for task_id, loader in self.val_loaders.items()
        }

    # ── Full Training Loop ────────────────────────────────────────────────

    def run(self, epochs: Optional[int] = None) -> Dict:
        """
        Run full training loop with early stopping.

        Returns
        -------
        dict with history, best_avg_auc, final_task_aucs
        """
        epochs = epochs or self.cfg.get("epochs", 200)
        self._resume_path = self.ckpt_path.replace(".pt", "_resume.pt")

        print(f"\n  Device: {self.device}")
        print(f"  PCGrad: {self.use_pcgrad}")
        print(f"  Epochs: {epochs} | Patience: {self.patience}")

        for epoch in range(self.start_epoch, epochs + 1):
            # ── Train ────────────────────────────────────────────────
            train_loss, pcgrad_stats = self.train_epoch()

            # ── Evaluate ─────────────────────────────────────────────
            task_aucs = self.eval_all_tasks()
            valid_aucs = [v for v in task_aucs.values() if v == v]  # Remove NaN
            avg_auc = np.mean(valid_aucs) if valid_aucs else 0.0

            # ── LR Schedule ──────────────────────────────────────────
            self.scheduler.step(avg_auc)

            # ── Early Stopping + Checkpoint ──────────────────────────
            if avg_auc > self.best_avg_auc + 1e-4:
                self.best_avg_auc = avg_auc
                self.patience_counter = 0
                self._save_checkpoint(epoch)
            else:
                self.patience_counter += 1

            # ── History ──────────────────────────────────────────────
            self.history["train_loss"].append(float(train_loss))
            self.history["val_auc_per_task"].append(
                {str(k): v for k, v in task_aucs.items()}
            )
            self.history["val_auc_avg"].append(float(avg_auc))

            if pcgrad_stats:
                self.history["pcgrad_stats"].append(pcgrad_stats)

            # ── Per-epoch resume checkpoint ───────────────────────────
            if epoch % 10 == 0 or self.patience_counter >= self.patience:
                torch.save({
                    "model": self.model.state_dict(),
                    "optimizer": self._base_optimizer.state_dict(),
                    "best_avg_auc": self.best_avg_auc,
                    "patience_counter": self.patience_counter,
                    "epoch": epoch,
                    "history": self.history,
                }, self._resume_path)

            # ── Logging ──────────────────────────────────────────────
            log_line = (
                f"Epoch {epoch:3d}/{epochs} | "
                f"Loss {train_loss:.4f} | "
                f"Val AUC {avg_auc:.4f} | "
                f"Best {self.best_avg_auc:.4f} | "
                f"Patience {self.patience_counter}/{self.patience}"
            )

            if pcgrad_stats:
                log_line += (
                    f" | Conflicts {pcgrad_stats.get('conflict_ratio_before', 0):.2%}"
                    f"→{pcgrad_stats.get('conflict_ratio_before', 0) - pcgrad_stats.get('conflict_reduction', 0):.2%}"
                )

            print(log_line)

            # ── Stop ─────────────────────────────────────────────────
            if self.patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

        # ── Load Best Model ───────────────────────────────────────────
        if os.path.exists(self.ckpt_path):
            state = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
            # Handle both full checkpoint (dict) and legacy (state_dict only)
            if isinstance(state, dict) and "model" in state:
                self.model.load_state_dict(state["model"])
            else:
                self.model.load_state_dict(state)

        # ── Final Eval ────────────────────────────────────────────────
        final_aucs = self.eval_all_tasks()

        return {
            "history": self.history,
            "best_avg_auc": self.best_avg_auc,
            "final_task_aucs": final_aucs,
        }