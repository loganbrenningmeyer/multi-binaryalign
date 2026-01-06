import os
from pathlib import Path
import argparse
import wandb
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from omegaconf import OmegaConf, DictConfig

from binaryalign.models import BinaryAlignClassifier, BinaryAlignModel, load_backbone
from binaryalign.data import BinaryAlignDataset, BinaryAlignCollator
from binaryalign.tokenization import BinaryAlignTokenizer
from binaryalign.training.trainer import Trainer


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
            entity=os.environ.get("WANDB_ENTITY", None)
        )
    except Exception as e:
        # -- Use offline if init fails
        print(f"---- wandb.init() failed, running offline: {e}")
        wandb.init(
            name=run_name,
            mode='offline'
        )

import random
import numpy as np

def measure_lengths(dataset, tokenizer, n=2000):
    lens = []
    for _ in range(n):
        ex = dataset[0]  # idx ignored if sampling dataset
        src_words = ex["src_words"]
        tgt_words = ex["tgt_words"]
        i = ex["src_word_idx"]

        src_marked = src_words[:i] + ["<ws>", src_words[i], "</ws>"] + src_words[i+1:]

        enc = tokenizer.tokenizer(
            src_marked,
            tgt_words,
            is_split_into_words=True,
            padding=False,
            truncation=False,   # IMPORTANT
            return_attention_mask=False,
        )
        lens.append(len(enc["input_ids"]))
    return np.array(lens)


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
    tokenizer = BinaryAlignTokenizer(model_name=config.model.backbone, max_length=config.model.max_length)
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
    # Create optimizer
    # ----------
    # -- Initialize optimizer with pre-training learning rate
    optimizer = optim.AdamW(model.parameters(), lr=config.train.pretrain.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * config.train.pretrain.steps),
        num_training_steps=config.train.pretrain.steps
    )

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
    
    finetune_tgt_lang = config.data.finetune.tgt_lang
    alpha = config.data.alpha

    # -- Create collator for DataLoader
    collator = BinaryAlignCollator(tokenizer)

    # -- Initialize Trainer w/ pre-training optimizer setup
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_dir=train_dir,
        logging_config=config.logging,
        start_step=start_step
    )

    # ====================
    # Pre-training
    # ====================
    if config.train.pretrain.steps > 0:
        # -- Create pre-training dataset
        pretrain_train_dataset = BinaryAlignDataset(
            train_manifest, finetune_tgt_lang, is_finetune=False, alpha=alpha, sample_with_replacement=True
        )
        pretrain_valid_dataset = BinaryAlignDataset(
            valid_manifest, finetune_tgt_lang, is_finetune=False, alpha=alpha, sample_with_replacement=False
        )

        lens = measure_lengths(pretrain_train_dataset, tokenizer)
        print("p50", np.percentile(lens, 50))
        print("p90", np.percentile(lens, 90))
        print("p95", np.percentile(lens, 95))
        print("p99", np.percentile(lens, 99))
        print("% > 256", (lens > 256).mean())
        print("% > 512", (lens > 512).mean())

        return

        pretrain_train_loader = DataLoader(
            pretrain_train_dataset, 
            batch_size=config.data.batch_size, 
            shuffle=False,
            collate_fn=collator,
            num_workers=4
        )
        pretrain_valid_loader = DataLoader(
            pretrain_valid_dataset, 
            batch_size=config.data.batch_size, 
            shuffle=False,
            collate_fn=collator,
            num_workers=4
        )

        trainer.train(
            train_loader=pretrain_train_loader, 
            valid_loader=pretrain_valid_loader,
            steps=config.train.pretrain.steps,
            stage="pretrain"
        )

    # ====================
    # Fine-tuning
    # ====================
    if config.train.finetune.steps > 0:
        # -- Create fine-tuning dataset
        finetune_train_dataset = BinaryAlignDataset(
            train_manifest, finetune_tgt_lang, is_finetune=True, alpha=0, sample_with_replacement=True
        )
        finetune_valid_dataset = BinaryAlignDataset(
            valid_manifest, finetune_tgt_lang, is_finetune=True, alpha=0, sample_with_replacement=False
        )

        finetune_train_loader = DataLoader(
            finetune_train_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=4
        )
        finetune_valid_loader = DataLoader(
            finetune_valid_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=4
        )

        # ----------
        # Update optimizer / scheduler for fine-tuning
        # ----------
        for pg in optimizer.param_groups:
            pg["lr"] = config.train.finetune.lr

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.10 * config.train.finetune.steps),
            num_training_steps=config.train.finetune.steps
        )
        trainer.scheduler = scheduler

        trainer.train(
            train_loader=finetune_train_loader, 
            valid_loader=finetune_valid_loader,
            steps=config.train.finetune.steps,
            stage="finetune"
        )



if __name__ == "__main__":
    main()