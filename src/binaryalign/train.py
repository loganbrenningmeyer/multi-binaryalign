import os
from pathlib import Path
import argparse
import wandb
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
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

    # ----------
    # Resume training
    # ----------
    if config.run.resume.enable:
        ckpt_path = train_dir / "checkpoints" / config.run.resume.ckpt_name
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # -- Load BinaryAlignModel / optimizer
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

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
    manifest_path = config.data.manifest_path
    finetune_tgt_lang = config.data.finetune.tgt_lang
    alpha = config.data.alpha

    # -- Create collator for DataLoader
    collator = BinaryAlignCollator(tokenizer)

    # -- Initialize Trainer w/ pre-training optimizer setup
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
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
        pretrain_dataset = BinaryAlignDataset(manifest_path, finetune_tgt_lang, is_finetune=False, alpha=alpha)

        pretrain_loader = DataLoader(
            pretrain_dataset, 
            batch_size=config.data.batch_size, 
            shuffle=False,
            collate_fn=collator,
            num_workers=4
        )

        trainer.train(
            loader=pretrain_loader, 
            steps=config.train.pretrain.steps,
            stage="pretrain"
        )

    # ====================
    # Fine-tuning
    # ====================
    if config.train.finetune.steps > 0:
        # -- Create fine-tuning dataset
        finetune_dataset = BinaryAlignDataset(manifest_path, finetune_tgt_lang, is_finetune=True, alpha=0)

        finetune_loader = DataLoader(
            finetune_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=4
        )

        # ----------
        # Update optimizer to fine-tuning learning rate
        # ----------
        for pg in optimizer.param_groups:
            pg["lr"] = config.train.finetune.lr

        trainer.train(
            loader=finetune_loader, 
            steps=config.train.finetune.steps,
            stage="finetune"
        )



if __name__ == "__main__":
    main()