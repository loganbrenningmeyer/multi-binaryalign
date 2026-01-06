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
        logging_config: DictConfig,
        start_step: int=1
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.device = device
        self.train_dir = train_dir

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
        train_step = 1

        while train_step <= steps:

            epoch_loss = 0.0
            num_batches = 0

            for batch in tqdm(train_loader, desc=f"({stage}) Epoch {epoch}"):
                if train_step > steps:
                    break

                # -- Perform training step
                loss = self.train_step(batch)

                # -- Log batch loss / save checkpoint
                self.log_loss(loss, self.global_step, "batch", stage)

                if self.global_step % self.ckpt_steps[stage] == 0:
                    self.save_checkpoint(self.global_step, stage)

                # -- Test on validation dataset
                if self.global_step % self.valid_steps == 0:
                    valid_loss = self.validate(valid_loader)
                    self.log_loss(valid_loss, self.global_step, "valid", stage)

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

        return loss.item()
    
    @torch.no_grad()
    def validate(self, valid_loader: DataLoader):
        """
        
        """
        self.model.eval()

        valid_loss = 0.0
        num_batches = 0

        for batch in tqdm(valid_loader, desc=f"Validation"):
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

        valid_loss /= num_batches

        return valid_loss


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
            "step": step,
            "stage": stage
        }, ckpt_path)