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
from binaryalign.data.utils.prepare import load_hansards, save_alignment_jsonl, load_alignment_jsonl


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
    hidden_dim = backbone.config.hidden_size
    classifier = BinaryAlignClassifier(hidden_dim)

    model = BinaryAlignModel(backbone, classifier)
    
    # ----------
    # Create DataLoader
    # ----------
    json_path = "datasets/en-fr/Hansards/test/alignments.jsonl"
    src_sentences, tgt_sentences, alignments = load_alignment_jsonl(json_path)

    dataset = BinaryAlignDataset(src_sentences, tgt_sentences, alignments)
    collator = BinaryAlignCollator(tokenizer)

    dataloader = DataLoader(
        dataset, 
        batch_size=config.data.batch_size, 
        shuffle=False,
        collate_fn=collator
    )

    # ----------
    # Define optimizer
    # ----------
    optimizer = optim.AdamW(model.parameters())

    # ----------
    # Create Trainer
    # ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        train_loader=dataloader,
        train_dir=train_dir,
        logging_config=config.logging
    )
    trainer.train(config.train.epochs)


if __name__ == "__main__":
    main()