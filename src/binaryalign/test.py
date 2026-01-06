import os
from pathlib import Path
import argparse
import spacy
import torch
from omegaconf import OmegaConf, DictConfig

from binaryalign.models import BinaryAlignClassifier, BinaryAlignModel, load_backbone
from binaryalign.tokenization import BinaryAlignTokenizer


def load_config(config_path: str) -> DictConfig:
    config = OmegaConf.load(config_path)
    return config

def save_config(config: DictConfig, save_path: str):
    OmegaConf.save(config, save_path)


def main():
    # ----------
    # Parse arguments / load config
    # ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    test_config = load_config(args.config)

    train_dir = Path(test_config.run.run_dir) / "training"
    train_config = load_config(train_dir / "config.yml")

    # ---------
    # Create Testing Dirs / Save Config
    # ----------
    test_dir = Path(test_config.run.run_dir) / "testing" / test_config.run.name
    os.makedirs(test_dir, exist_ok=True)

    save_config(test_config, test_dir / "config.yml")

    # ----------
    # Load tokenizer/backbone
    # ----------
    tokenizer = BinaryAlignTokenizer(model_name=train_config.model.backbone, max_length=train_config.model.max_length)
    backbone = load_backbone(train_config.model.backbone, tokenizer.vocab_size)

    # ----------
    # Load BinaryAlignModel checkpoint
    # ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = backbone.config.hidden_size

    classifier = BinaryAlignClassifier(hidden_dim)
    model = BinaryAlignModel(backbone, classifier)

    ckpt_path = train_dir / "checkpoints" / test_config.run.checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # ----------
    # Run inference on test samples
    # ----------
    threshold = test_config.inference.threshold

    src_lang = test_config.inference.src.lang
    src_sentences = test_config.inference.src.sentences

    tgt_lang = test_config.inference.tgt.lang
    tgt_sentences = test_config.inference.tgt.sentences

    src_nlp = spacy.blank(src_lang)
    tgt_nlp = spacy.blank(tgt_lang)

    for src_sent, tgt_sent in zip(src_sentences, tgt_sentences):
        # -- Break sentences into words
        src_words = [t.text for t in src_nlp(src_sent)]
        tgt_words = [t.text for t in tgt_nlp(tgt_sent)]
        # -- Align for all source words
        src_idxs = list(range(len(src_words)))
        # -- Form batch for each source word
        src_batch = [src_words] * len(src_idxs)
        tgt_batch = [tgt_words] * len(src_idxs)

        encoding = tokenizer.encode_marked_batch(src_batch, tgt_batch, src_idxs)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        target_mask = (encoding["token_type_ids"] == 1)

        # -- Run inference
        preds = model.predict(input_ids, attention_mask, target_mask, threshold)

        print(f"preds: {preds}")



if __name__ == "__main__":
    main()