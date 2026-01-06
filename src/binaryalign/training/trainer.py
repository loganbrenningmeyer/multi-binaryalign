import os
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
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
        device: torch.device,
        train_dir: str,
        logging_config: DictConfig
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device
        self.train_dir = train_dir

        # -- Logging parameters
        self.global_step = 1
        self.wandb_enabled = logging_config.wandb.enable
        self.wandb_save_ckpt = logging_config.wandb.save_ckpt
        self.loss_steps = logging_config.loss_steps
        self.valid_steps = logging_config.valid_steps
        self.ckpt_steps = logging_config.ckpt_steps

    def train(self, loader: DataLoader, steps: int, stage: str="pretrain"):
        """
        Trains the BinaryAlignModel for the specified number of epochs.

        Parameters:
            epochs (int): Total number of training epochs
        """
        self.model.train()

        epoch = 1
        train_step = 1

        while train_step < steps:

            epoch_loss = 0.0
            num_batches = 0

            for batch in tqdm(loader, desc=f"({stage}) Epoch {epoch}"):
                if train_step >= steps:
                    break

                # -- Perform training step
                loss = self.train_step(batch)

                # -- Log batch loss / save checkpoint
                self.log_loss(loss, self.global_step, "batch", stage)

                if self.global_step % self.ckpt_steps[stage] == 0:
                    self.save_checkpoint(self.global_step, stage)

                epoch_loss += loss
                num_batches += 1
                self.global_step += 1
                train_step += 1

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
        labels_target = batch["labels_target"].to(self.device)

        # ----------
        # Forward pass
        # ----------
        logits_target = self.model(input_ids, attention_mask, target_mask)

        # ----------
        # Compute loss / update
        # ----------
        loss = self.criterion(logits_target, labels_target)
        loss.backward()
        self.optimizer.step()

        return loss.item()

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
            "step": step
        }, ckpt_path)