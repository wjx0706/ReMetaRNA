import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from tqdm.auto import tqdm

from modeling_rna import RNAMultitaskModel


@dataclass
class TrainingArguments:
    """Configuration for the training."""

    # training parameter
    num_epochs: int = 100
    optimizer_type: str = "adam"  # "sgd" or "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    adam_beta1: float = 0.9  # also: momentum for SGD
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-08
    # training loaders
    training_batch_size: int = 32
    eval_batch_size: int = 32
    test_batch_size: int = 32
    num_workers: int = 4
    # learning rate scheduler
    use_cos_scheduler: bool = True
    warmup_ratio: float = 0.1
    warmup_init_lr: float = 1e-6
    min_lr: float = 1e-7
    # swap prediction parameters
    temperature: float = 0.1
    sinkhorn_eps: float = 0.05
    sinkhorn_iters: int = 3
    freeze_prototypes_nepochs: int = 0
    # other
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = False
    checkpoint_dir: str | None = None
    save_best_model: bool = True
    # logging
    log_level: str = "INFO"


class Trainer:
    def __init__(
        self,
        model: RNAMultitaskModel,
        training_args: TrainingArguments,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ) -> None:
        self.config = training_args
        self.device = training_args.device
        self.set_seed(training_args.seed)

        model.reset_parameters()
        self.model = model
        # move model to device
        self.model.to(self.device)

        # create checkpoint directory
        self._setup_checkpoint_dir()
        # setup logging
        self.logger = self._setup_logger()

        # prepare dataloader
        if train_dataset is not None:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=training_args.training_batch_size,
                shuffle=True,
                num_workers=training_args.num_workers,
                pin_memory=True,
            )
        else:
            self.train_loader = None

        if eval_dataset is not None:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=training_args.eval_batch_size,
                shuffle=False,
            )
        else:
            self.eval_loader = None

        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=training_args.test_batch_size,
                shuffle=False,
            )
        else:
            self.test_loader = None

        if self.train_loader:  # only setup optimizer and scheduler if training
            # setup optimizer
            self.optimizer = self._setup_optimizer()

            # setup scheduler
            total_steps = len(self.train_loader) * training_args.num_epochs
            self.scheduler = None
            if training_args.use_cos_scheduler:
                self.scheduler = self._setup_scheduler(total_steps)
                self.optimizer = LARC(
                                self.optimizer,
                                trust_coefficient=0.001,
                                clip=False,
                            )

            # loss functions
            self.seq_loss_fn = nn.CrossEntropyLoss()
            self.swap_loss_fn = nn.CrossEntropyLoss()

            # AMP scaler
            self.scaler = torch.GradScaler(enabled=training_args.use_amp)

        # metrics
        self.best_recovery = 0

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        if self.config.optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=self.config.adam_beta1,
            )
        elif self.config.optimizer_type.lower() == "adam":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")

        return optimizer

    def _setup_scheduler(
        self, total_steps: int
    ) -> torch.optim.lr_scheduler.SequentialLR:
        warmup_steps = int(self.config.warmup_ratio * total_steps)
        max_lr = self.config.learning_rate

        warmup_scheduler = LinearLR(
            self.optimizer,  # type: ignore
            start_factor=self.config.warmup_init_lr / max_lr,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,  # type: ignore
            T_max=total_steps - warmup_steps, 
            eta_min=self.config.min_lr
        )

        return SequentialLR(
            self.optimizer,  # type: ignore
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    def _setup_checkpoint_dir(self) -> None:
        """Create checkpoint directory."""
        if self.config.checkpoint_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.config.checkpoint_dir = f"./checkpoints-{timestamp}"
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # save training arguments
        with open(
            os.path.join(self.config.checkpoint_dir, "training_args.json"), "w"
        ) as f:
            json.dump(self.config.__dict__, f, indent=4)

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the trainer class."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Remove any existing handlers
        if logger.handlers:
            for handler in logger.handlers:
                logger.removeHandler(handler)

        assert self.config.checkpoint_dir is not None, "Checkpoint directory is not set"

        # Create file handler that logs to a file in the checkpoint directory
        log_file = os.path.join(self.config.checkpoint_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Create console handler that logs to stdout without timestamps
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        return logger

    def plot_training_curves(self, train_seq_losses, eval_seq_losses) -> None:
        
        assert self.config.checkpoint_dir is not None, "Checkpoint directory is not set"
        
        # Plot losses
        epochs = range(1, len(train_seq_losses) + 1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_seq_losses, 'b-', label='Train Seq Loss')
        ax.plot(epochs, eval_seq_losses, 'r-', label='Eval Seq Loss')
        
        ax.set_title('Training and Evaluation Sequence Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        # Save plot
        save_path = os.path.join(self.config.checkpoint_dir, "loss_curve.png")
        fig.savefig(save_path)

        
    def save_model(
        self,
        name: str,
        epoch: int,
        train_loss: tuple,
        eval_loss: tuple,
        recovery: float,
    ) -> None:
        """Save model weights and config information."""
        model_path = f"{name}.pt"
        torch.save(self.model.state_dict(), model_path)

        metadata = {
            "config": self.config.__dict__,
            "metrics": {
                "current_recovery": recovery,
                "best_recovery": self.best_recovery,
                "train_loss": train_loss[0],
                "train_swap_loss": train_loss[1],
                "train_seq_loss": train_loss[2],
                "eval_loss": eval_loss[0],
                "eval_swap_loss": eval_loss[1],
                "eval_seq_loss": eval_loss[2],
                "current_epoch": epoch,
                "total_epochs": self.config.num_epochs,
            },
        }

        metadata_path = f"{name}_info.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def load_model(self, model_path: str) -> None:
        """Load model weights."""
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)

    def get_offset(
        self, local_nucl_ids: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Get offset to turn local_nucl_ids into global nucl_ids in a batch.
        """
        num_nucl_per_rna = scatter(local_nucl_ids + 1, batch, dim=0, reduce="max")
        return torch.cumsum(
            torch.cat([local_nucl_ids.new_tensor([0]), num_nucl_per_rna]), dim=0
        )

    def train_epoch(self, epoch: int) -> tuple[float, float, float]:
        """Train for one epoch."""
        assert self.train_loader is not None, "No training dataset provided"

        self.model.train()
        epoch_loss = 0.0
        epoch_swap_loss = 0.0
        epoch_seq_loss = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f"Training {epoch}/{self.config.num_epochs}",
            leave=False,
        )

        for data_batch, aug_data_batch in pbar:
            self.optimizer.zero_grad()

            # move data to device
            x_s = data_batch.x.to(self.device)
            x_t = aug_data_batch.x.to(self.device)
            edge_index = data_batch.edge_index.to(self.device)
            batch = data_batch.batch.to(self.device)
            edge_attr = data_batch.edge_attr.to(self.device)
            atom_ids = data_batch.atom_ids.to(self.device)
            seq = data_batch.seq.to(self.device).repeat(2)
            local_nucl_ids = data_batch.nucl_ids.to(self.device)
            nucl_offsets = self.get_offset(local_nucl_ids, batch)
            batch_nucl_ids = local_nucl_ids + nucl_offsets[batch]
            batch_nucl_ids = batch_nucl_ids.to(self.device)

            # forward pass
            with torch.autocast(device_type=self.device, enabled=self.config.use_amp):
                logits, scores_s, scores_t = self.model(
                    x_s, x_t, edge_attr, edge_index, batch, atom_ids, batch_nucl_ids
                )
                seq_loss = self.seq_loss_fn(logits, seq)

                # swap prediction
                with torch.no_grad():
                    q_s = sinkhorn(
                        scores_s,
                        eps=self.config.sinkhorn_eps,
                        num_iters=self.config.sinkhorn_iters,
                    )
                    q_t = sinkhorn(
                        scores_t,
                        eps=self.config.sinkhorn_eps,
                        num_iters=self.config.sinkhorn_iters,
                    )

                swap_loss = 0.5 * (
                    self.swap_loss_fn(scores_s / self.config.temperature, q_t)
                    + self.swap_loss_fn(scores_t / self.config.temperature, q_s)
                )
                # alpha = 0.5
                # loss = (1 - alpha) * seq_loss + alpha * swap_loss # / swap_loss.item() * seq_loss.item()
                loss = seq_loss

            # backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            if epoch < self.config.freeze_prototypes_nepochs:
                for name, p in self.model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None

            self.scaler.step(self.optimizer)  # type: ignore
            self.scaler.update()

            # update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # normalize prototypes
            with torch.no_grad():
                w = self.model.prototypes.weight
                w = nn.functional.normalize(w, p=2, dim=1)
                self.model.prototypes.weight.copy_(w)

            pbar.set_postfix(seq_loss=seq_loss.item(), swap_loss=swap_loss.item())
            epoch_swap_loss += swap_loss.item()
            epoch_seq_loss += seq_loss.item()
            epoch_loss += loss.item()

            avg_epoch_swap_loss = epoch_swap_loss / len(self.train_loader)
            avg_epoch_seq_loss = epoch_seq_loss / len(self.train_loader)
            avg_epoch_loss = epoch_loss / len(self.train_loader)

        return avg_epoch_loss, avg_epoch_swap_loss, avg_epoch_seq_loss

    def evaluate(self) -> tuple[tuple[float, float, float], float]:
        """Evaluate the model."""
        assert self.eval_loader is not None, "No evaluation dataset provided"

        self.model.eval()
        total_swap_loss = 0.0
        total_seq_loss = 0.0
        total_recovery = 0.0
        num_seqs = 0

        with torch.no_grad():
            pbar = tqdm(self.eval_loader, desc="Evaluating", leave=False)
            for data_batch, aug_data_batch in pbar:
                # Move data to device
                x_s = data_batch.x.to(self.device)
                x_t = aug_data_batch.x.to(self.device)
                edge_index = data_batch.edge_index.to(self.device)
                batch = data_batch.batch.to(self.device)
                edge_attr = data_batch.edge_attr.to(self.device)
                atom_ids = data_batch.atom_ids.to(self.device)
                seq = data_batch.seq.to(self.device).repeat(2)
                local_nucl_ids = data_batch.nucl_ids.to(self.device)
                nucl_offsets = self.get_offset(local_nucl_ids, batch)
                batch_nucl_ids = local_nucl_ids + nucl_offsets[batch]
                batch_nucl_ids = batch_nucl_ids.to(self.device)

                # for recovery computing
                nucl_offsets = nucl_offsets.to(self.device)
                nucl_offsets = torch.concat(
                    [nucl_offsets[:-1], nucl_offsets + nucl_offsets[-1]], dim=0
                )

                logits, scores_s, scores_t = self.model(
                    x_s, x_t, edge_attr, edge_index, batch, atom_ids, batch_nucl_ids
                )

                seq_loss = self.seq_loss_fn(logits, seq)

                q_s = sinkhorn(
                    scores_s,
                    eps=self.config.sinkhorn_eps,
                    num_iters=self.config.sinkhorn_iters,
                )
                q_t = sinkhorn(
                    scores_t,
                    eps=self.config.sinkhorn_eps,
                    num_iters=self.config.sinkhorn_iters,
                )

                swap_loss = 0.5 * (
                    self.swap_loss_fn(scores_s / self.config.temperature, q_t)
                    + self.swap_loss_fn(scores_t / self.config.temperature, q_s)
                )

                recovery = compute_recovery(logits, seq, nucl_offsets)

                total_swap_loss += swap_loss.item()
                total_seq_loss += seq_loss.item()
            
                total_recovery += sum(recovery)
                num_seqs += len(nucl_offsets) - 1

        avg_swap_loss = total_swap_loss / len(self.eval_loader)
        avg_seq_loss = total_seq_loss / len(self.eval_loader)
        avg_loss = avg_swap_loss + avg_seq_loss
        avg_recovery = total_recovery / num_seqs

        return (avg_loss, avg_swap_loss, avg_seq_loss), avg_recovery

    def train(self) -> tuple[list[float], list[float], list[float]]:
        assert self.config.checkpoint_dir is not None, "Checkpoint directory is not set"

        train_losses = []
        eval_losses = []
        recoveries = []

        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(epoch + 1)
            train_losses.append(train_loss)

            eval_loss, recovery = self.evaluate()
            eval_losses.append(eval_loss)
            recoveries.append(recovery)

            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs}\n"
                f"    Train Loss: {train_loss[0]:.4f}\t"
                f"Swap Loss: {train_loss[1]:.4f}, Seq Loss: {train_loss[2]:.4f}\n"
                f"    Eval Loss: {eval_loss[0]:.4f}\t"
                f"Swap Loss: {eval_loss[1]:.4f}, Seq Loss: {eval_loss[2]:.4f}\n"
                f"    Recovery: {recovery:.4f}"
            )

            # save best model
            if recovery > self.best_recovery:
                self.best_recovery = recovery
                if self.config.save_best_model:
                    self.save_model(
                        os.path.join(self.config.checkpoint_dir, "best_model"),
                        epoch=epoch,
                        train_loss=train_loss,
                        eval_loss=eval_loss,
                        recovery=recovery,
                    )
                    self.logger.info(f"Save best model at epoch: {epoch}")

        # save final model
        self.save_model(
            os.path.join(self.config.checkpoint_dir, "final_model"),
            epoch=epoch,
            train_loss=train_loss,
            eval_loss=eval_loss,
            recovery=recovery,
        )

        self.plot_training_curves(
            train_losses,
            eval_losses,
        )
        
        return train_losses, eval_losses, recoveries

    def test(self):
        assert self.test_loader is not None, "No test dataset provided"

        self.model.eval()
        num_seqs = 0
        recoveries = []
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing", leave=False)
            for data_batch in pbar:
                x = data_batch.x.to(self.device)
                edge_index = data_batch.edge_index.to(self.device)
                batch = data_batch.batch.to(self.device)
                edge_attr = data_batch.edge_attr.to(self.device)
                atom_ids = data_batch.atom_ids.to(self.device)
                seq = data_batch.seq.to(self.device)
                local_nucl_ids = data_batch.nucl_ids.to(self.device)
                nucl_offsets = self.get_offset(local_nucl_ids, batch)
                batch_nucl_ids = local_nucl_ids + nucl_offsets[batch]
                batch_nucl_ids = batch_nucl_ids.to(self.device)

                # for recovery computing
                nucl_offsets = nucl_offsets.to(self.device)
                # nucl_offsets = torch.concat(
                #     [nucl_offsets[:-1], nucl_offsets + nucl_offsets[-1]], dim=0
                # )

                logits = self.model(
                    x, edge_attr, edge_index, batch, atom_ids, batch_nucl_ids
                )
                # print(logits.shape, seq.shape)
                recovery = compute_recovery(logits, seq, nucl_offsets)
                num_seqs += len(nucl_offsets) - 1
                recoveries.extend(recovery)
                
        avg_recovery = sum(recoveries) / num_seqs
        print(f"Test Recovery: {avg_recovery:.4f}")

        return avg_recovery, recoveries


def sinkhorn(
    scores: torch.Tensor, eps: float = 0.05, num_iters: int = 3
) -> torch.Tensor:
    device = scores.device

    Q = torch.exp(scores / eps)
    Q /= Q.sum()
    B, K = Q.shape
    u = torch.zeros(K, device=device)
    c, r = torch.ones(K, device=device) / K, torch.ones(B, device=device) / B
    for _ in range(num_iters):
        # column normalization
        u = Q.sum(dim=0)  # [K]
        Q *= (c / u).unsqueeze(0)  # [K] -> [1, K] -> [B, K]
        # row normalization
        Q *= (r / Q.sum(dim=1)).unsqueeze(1)  # [B] -> [B, 1] -> [B, K]

    # row normalization
    # return the probability of each sample to be assigned to each prototype
    return Q / Q.sum(dim=1, keepdim=True)  # [B, K]


def compute_recovery(
    logits: torch.Tensor, seq: torch.Tensor, ptr: torch.Tensor
) -> list[float]:
    """
    Compute sequence recovery rate for a mini-batch。

    Return:
        - recoveries: list of recovery rates for each RNA sequence in the batch.
    """
    recoveries = []
    predicted_seq = torch.argmax(logits, dim=1)

    for i in range(len(ptr) - 1):
        start_idx, end_idx = ptr[i], ptr[i + 1]
        recovery = (
            (predicted_seq[start_idx:end_idx] == seq[start_idx:end_idx]).float().mean()
        )
        recoveries.append(recovery.detach().cpu().item())

    return recoveries


# copy from nvidia apex
class LARC(object):
    """
    :class:`LARC` is a pytorch implementation of both the scaling and clipping variants of LARC,
    in which the ratio between gradient and parameter magnitudes is used to calculate an adaptive
    local learning rate for each individual parameter. The algorithm is designed to improve
    convergence of large batch training.

    See https://arxiv.org/abs/1708.03888 for calculation of the local learning rate.

    In practice it modifies the gradients of parameters as a proxy for modifying the learning rate
    of the parameters. This design allows it to be used as a wrapper around any torch.optim Optimizer.

    ```
    model = ...
    optim = torch.optim.Adam(model.parameters(), lr=...)
    optim = LARC(optim)
    ```

    It can even be used in conjunction with apex.fp16_utils.FP16_optimizer.

    ```
    model = ...
    optim = torch.optim.Adam(model.parameters(), lr=...)
    optim = LARC(optim)
    optim = apex.fp16_utils.FP16_Optimizer(optim)
    ```

    Args:
        optimizer: Pytorch optimizer to wrap and modify learning rate for.
        trust_coefficient: Trust coefficient for calculating the lr. See https://arxiv.org/abs/1708.03888
        clip: Decides between clipping or scaling mode of LARC. If `clip=True` the learning rate is set to `min(optimizer_lr, local_lr)` for each parameter. If `clip=False` the learning rate is set to `local_lr*optimizer_lr`.
        eps: epsilon kludge to help with numerical stability while calculating adaptive_lr
    """

    def __init__(self, optimizer, trust_coefficient=0.02, clip=True, eps=1e-8):
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    @property
    def state(self):
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group["weight_decay"] if "weight_decay" in group else 0
                weight_decays.append(weight_decay)
                group["weight_decay"] = 0
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        # calculate adaptive lr + weight decay
                        adaptive_lr = (
                            self.trust_coefficient
                            * (param_norm)
                            / (grad_norm + param_norm * weight_decay + self.eps)
                        )

                        # clip learning rate for LARC
                        if self.clip:
                            # calculation of adaptive_lr so that when multiplied by lr it equals `min(adaptive_lr, lr)`
                            adaptive_lr = min(adaptive_lr / group["lr"], 1)

                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group["weight_decay"] = weight_decays[i]
