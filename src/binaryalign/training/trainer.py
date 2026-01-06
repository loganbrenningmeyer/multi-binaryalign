import os
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from omegaconf import DictConfig
from tqdm import tqdm

from binaryalign.models import BinaryAlignModel


class Trainer:
    """


    Parameters:

    """

    def __init__(
        self,
        model: BinaryAlignModel,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        device: torch.device,
        train_dir: str,
        logging_config: DictConfig,
        threshold: float=0.5,
        start_step: int=1
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.device = device
        self.train_dir = train_dir
        self.threshold = threshold

        # -- Logging parameters
        self.global_step = start_step
        self.wandb_enabled = logging_config.wandb.enable
        self.wandb_save_ckpt = logging_config.wandb.save_ckpt
        self.loss_steps = logging_config.loss_steps
        self.valid_steps = logging_config.valid_steps
        self.ckpt_steps = logging_config.ckpt_steps

    def train(
        self, 
        train_loader: DataLoader, 
        valid_loader: DataLoader, 
        steps: int, 
        stage: str="pretrain"
    ):
        """
        Trains the BinaryAlignModel for the specified number of epochs.

        Parameters:
            epochs (int): Total number of training epochs
        """
        self.model.train()

        epoch = 1

        while self.global_step <= steps:

            epoch_loss = 0.0
            num_batches = 0

            for batch in tqdm(train_loader, desc=f"({stage}) Epoch {epoch}"):
                if self.global_step > steps:
                    break

                # -- Perform training step
                loss = self.train_step(batch)

                # -- Log batch loss / save checkpoint
                self.log_loss(loss, self.global_step, "batch", stage)

                if self.global_step % self.ckpt_steps[stage] == 0:
                    self.save_checkpoint(self.global_step, stage)

                # -- Test on validation dataset
                if self.global_step % self.valid_steps == 0:
                    metrics = self.validate(valid_loader)
                    self.log_valid(metrics, self.global_step, stage)

                epoch_loss += loss
                num_batches += 1
                self.global_step += 1

            # -- Log epoch loss
            epoch_loss /= num_batches
            self.log_loss(epoch_loss, self.global_step - 1, "epoch", stage)
            epoch += 1

    def train_step(self, batch: dict):
        """
        
        """
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        target_mask = batch["target_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # ----------
        # Forward pass
        # ----------
        logits = self.model(input_ids, attention_mask)

        # ----------
        # Compute loss / mask padding & src / average
        # ----------
        loss_per_token = self.criterion(logits, labels)
        mask = target_mask & attention_mask.bool()
        loss = loss_per_token[mask].mean()

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()
    
    @torch.no_grad()
    def validate(self, valid_loader: DataLoader):
        """
        
        """
        self.model.eval()

        valid_loss = 0.0
        num_batches = 0

        tp = fp = fn = tn = 0

        for batch in tqdm(valid_loader, desc=f"Validation (step {self.global_step})"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_mask = batch["target_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # ----------
            # Forward pass
            # ----------
            logits = self.model(input_ids, attention_mask)

            # ----------
            # Compute loss / mask padding & src / average
            # ----------
            loss_per_token = self.criterion(logits, labels)
            mask = target_mask & attention_mask.bool()
            loss = loss_per_token[mask].mean()

            valid_loss += loss.item()
            num_batches += 1

            # ----------
            # Compute TP/FP/FN/TN
            # ----------
            scores = torch.sigmoid(logits)
            preds = scores >= self.threshold

            y_true = labels[mask].bool()
            y_pred = preds[mask]

            tp += int((y_pred & y_true).sum().item())
            fp += int((y_pred & ~y_true).sum().item())
            fn += int((~y_pred & y_true).sum().item())
            tn += int((~y_pred & ~y_true).sum().item())

        valid_loss /= num_batches

        eps = 1e-12
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        acc = (tp + tn) / (tp + tn + fp + fn + eps)

        return {
            "loss": valid_loss,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": acc
        }
    
    def log_valid(self, metrics: dict, step: int, stage: str):
        if self.wandb_enabled:
            wandb.log({f"{stage}/valid/loss": metrics["loss"]}, step=step)
            wandb.log({f"{stage}/valid/f1": metrics["f1"]}, step=step)
            wandb.log({f"{stage}/valid/precision": metrics["precision"]}, step=step)
            wandb.log({f"{stage}/valid/recall": metrics["recall"]}, step=step)
            wandb.log({f"{stage}/valid/accuracy": metrics["accuracy"]}, step=step)

    def log_loss(self, loss: float, step: int, label: str, stage: str):
        """
        Logs loss to wandb dashboard
        """
        if self.wandb_enabled:
            # -- Global loss
            wandb.log({f"global/{label}/loss": loss}, step=step)
            # -- Stage loss
            wandb.log({f"{stage}/{label}/loss": loss}, step=step)

    def save_checkpoint(self, step: int, stage: str):
        """
        
        """
        ckpt_path = os.path.join(self.train_dir, "checkpoints", f"model-{stage}-step{step}.ckpt")

        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": step,
            "stage": stage
        }, ckpt_path)