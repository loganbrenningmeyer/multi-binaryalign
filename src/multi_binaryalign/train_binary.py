import os
from pathlib import Path
import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from omegaconf import OmegaConf, DictConfig

from multi_binaryalign.models import (
    BinaryAlignClassifier,
    BinaryAlignModel,
    load_backbone,
)
from multi_binaryalign.data import BinaryAlignDataset, BinaryAlignCollator
from multi_binaryalign.tokenization import BinaryAlignTokenizer
from multi_binaryalign.training.binary_trainer import BinaryTrainer


def load_config(config_path: str) -> DictConfig:
    config = OmegaConf.load(config_path)
    return config


def save_config(config: DictConfig, save_path: str):
    OmegaConf.save(config, save_path)


def init_wandb(run_name: str):
    """
    Initializes wandb for logging, runs in offline mode on failure
    """
    try:
        wandb.init(
            name=run_name,
            project=os.environ.get("WANDB_PROJECT", "binaryalign"),
            entity=os.environ.get("WANDB_ENTITY", None),
        )
    except Exception as e:
        # -- Use offline if init fails
        print(f"---- wandb.init() failed, running offline: {e}")
        wandb.init(name=run_name, mode="offline")


def main():
    # ----------
    # Parse arguments / load config
    # ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    # ---------
    # Create Training Dirs / Save Config
    # ----------
    train_dir = Path(config.run.runs_dir) / config.run.name / "training"
    os.makedirs(train_dir / "checkpoints", exist_ok=True)

    save_config(config, train_dir / "config.yml")

    # ----------
    # Initialize wandb
    # ----------
    if config.logging.wandb.enable:
        init_wandb(config.run.name)

    # ----------
    # Load tokenizer/backbone
    # ----------
    tokenizer = BinaryAlignTokenizer(
        model_name=config.model.backbone, max_length=config.model.max_length
    )
    backbone = load_backbone(config.model.backbone, tokenizer.vocab_size)

    # ----------
    # Create classifier / BinaryAlignModel
    # ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = backbone.config.hidden_size
    classifier = BinaryAlignClassifier(hidden_dim)

    model = BinaryAlignModel(backbone, classifier)
    model.to(device)

    # ----------
    # Create optimizer / scheduler
    # ----------
    lr = config.train.pretrain.lr
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    lr_warmup = config.train.pretrain.lr_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(lr_warmup * config.train.pretrain.steps),
        num_training_steps=config.train.pretrain.steps,
    )

    # -------------------------
    # Create criterion (w/ or w/o pos_weight)
    # -------------------------
    # -- Enable pos_weight val from config
    if config.train.pos_weight.enable:
        criterion = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=torch.tensor(config.train.pos_weight.weight, device=device),
        )
    # -- No pos_weight
    else:
        criterion = nn.BCEWithLogitsLoss(reduction="none")

    # ----------
    # Resume training
    # ----------
    if config.run.resume.enable:
        ckpt_path = train_dir / "checkpoints" / config.run.resume.ckpt_name
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # -- Load BinaryAlignModel / optimizer
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

        # -- Move optimizer to GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        # -- Resume from last step
        start_step = ckpt["step"] + 1
    else:
        # -- If not resuming, start at step 1
        start_step = 1

    # ====================
    # Training
    # ====================
    train_manifest = config.data.train_manifest
    valid_manifest = config.data.valid_manifest

    finetune_tgt_lang = config.data.finetune_tgt_lang
    alpha = config.data.alpha

    # -- Create collator for DataLoader
    collator = BinaryAlignCollator(tokenizer)

    # -- Initialize Trainer w/ pre-training optimizer setup
    trainer = BinaryTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        train_dir=train_dir,
        logging_config=config.logging,
        start_step=start_step,
    )

    # ====================
    # Pre-training
    # ====================
    if config.train.pretrain.steps > 0:
        # -- Create pre-training dataset
        pretrain_train_dataset = BinaryAlignDataset(
            train_manifest,
            is_finetune=False,
            finetune_tgt_lang=finetune_tgt_lang,
            alpha=alpha,
            sample_with_replacement=True,
        )
        pretrain_valid_dataset = BinaryAlignDataset(
            valid_manifest,
            is_finetune=False,
            finetune_tgt_lang=finetune_tgt_lang,
            alpha=alpha,
            sample_with_replacement=False,
        )

        pretrain_train_loader = DataLoader(
            pretrain_train_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=4,
        )
        pretrain_valid_loader = DataLoader(
            pretrain_valid_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=4,
        )

        trainer.train(
            train_loader=pretrain_train_loader,
            valid_loader=pretrain_valid_loader,
            steps=config.train.pretrain.steps,
            stage="pretrain",
        )

    # ====================
    # Fine-tuning
    # ====================
    if config.train.finetune.steps > 0 and finetune_tgt_lang != None:
        # -- Create fine-tuning dataset
        finetune_train_dataset = BinaryAlignDataset(
            train_manifest,
            is_finetune=True,
            finetune_tgt_lang=finetune_tgt_lang,
            alpha=0,
            sample_with_replacement=True,
        )
        finetune_valid_dataset = BinaryAlignDataset(
            valid_manifest,
            is_finetune=True,
            finetune_tgt_lang=finetune_tgt_lang,
            alpha=0,
            sample_with_replacement=False,
        )

        finetune_train_loader = DataLoader(
            finetune_train_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=4,
        )
        finetune_valid_loader = DataLoader(
            finetune_valid_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=4,
        )

        # ----------
        # Update optimizer / scheduler for fine-tuning
        # ----------
        for pg in optimizer.param_groups:
            pg["lr"] = config.train.finetune.lr

        lr_warmup = config.train.finetune.lr_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(lr_warmup * config.train.finetune.steps),
            num_training_steps=config.train.finetune.steps,
        )
        trainer.scheduler = scheduler

        trainer.train(
            train_loader=finetune_train_loader,
            valid_loader=finetune_valid_loader,
            steps=config.train.finetune.steps,
            stage="finetune",
        )


if __name__ == "__main__":
    main()
