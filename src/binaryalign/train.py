import os
import argparse
import wandb
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
    train_dir = os.path.join(config.run.runs_dir, config.run.name, "training")
    os.makedirs(os.path.join(train_dir, 'checkpoints'), exist_ok=True)

    save_config(config, os.path.join(train_dir, 'config.yml'))

    # ----------
    # Initialize wandb
    # ----------
    if config.logging.wandb.enable:
        init_wandb(config.run.name)

    # ----------
    # Load tokenizer/backbone
    # ----------
    tokenizer = BinaryAlignTokenizer(config.model.backbone)
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
    # Create pretraining / finetuning BinaryAlignDatasets
    # ----------
    manifest_path = config.data.manifest_path
    finetune_tgt_lang = config.data.finetune.tgt_lang
    alpha = config.data.alpha

    pretrain_dataset = BinaryAlignDataset(manifest_path, finetune_tgt_lang, is_finetune=False, alpha=alpha)
    finetune_dataset = BinaryAlignDataset(manifest_path, finetune_tgt_lang, is_finetune=True, alpha=0)

    # ----------
    # Create pretraining / finetuning DataLoaders
    # ----------
    collator = BinaryAlignCollator(tokenizer)

    pretrain_loader = DataLoader(
        pretrain_dataset, 
        batch_size=config.data.batch_size, 
        shuffle=False,
        collate_fn=collator
    )

    finetune_loader = DataLoader(
        finetune_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collator
    )

    # ====================
    # Pre-training
    # ====================
    # -- Initialize optimizer with pre-training learning rate
    optimizer = optim.AdamW(model.parameters(), lr=config.train.pretrain.lr)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        train_dir=train_dir,
        logging_config=config.logging
    )

    trainer.train(
        loader=pretrain_loader, 
        steps=config.train.pretrain.steps,
        stage="pretrain"
    )

    # ====================
    # Fine-tuning
    # ====================
    # -- Set fine-tuning optimizer learning rate
    for pg in optimizer.param_groups:
        pg["lr"] = config.train.finetune.lr

    trainer.train(
        loader=finetune_loader, 
        steps=config.train.finetune.steps,
        stage="finetune"
    )



if __name__ == "__main__":
    main()