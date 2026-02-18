import os
import json
from pathlib import Path
import argparse
import spacy
import torch
from omegaconf import OmegaConf, DictConfig

from multi_binaryalign.models import (
    BinaryAlignClassifier,
    BinaryAlignModel,
    load_backbone,
)
from multi_binaryalign.tokenization import BinaryAlignTokenizer
from multi_binaryalign.inference.align import BinaryAlign
from multi_binaryalign.tokenization import Segmenter


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
    tokenizer = BinaryAlignTokenizer(
        model_name=train_config.model.backbone, max_length=train_config.model.max_length
    )
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
    # Create BinaryAlign inference model
    # ----------
    src_lang = test_config.inference.src.lang
    tgt_lang = test_config.inference.tgt.lang

    segmenter = Segmenter(src_lang, tgt_lang)
    binaryalign = BinaryAlign(model, tokenizer)

    # ----------
    # Run inference on test samples
    # ----------
    threshold = test_config.inference.threshold

    src_sentences = test_config.inference.src.sentences
    tgt_sentences = test_config.inference.tgt.sentences

    align_json = {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "threshold": threshold,
        "sentence_pairs": [],
    }

    for src_sent, tgt_sent in zip(src_sentences, tgt_sentences):
        src_words = segmenter.split_words(src_sent, src_lang)
        tgt_words = segmenter.split_words(tgt_sent, tgt_lang)

        src_alignments, tgt_alignments = binaryalign.align_sentence_pair(
            src_words, tgt_words, threshold
        )

        align_json["sentence_pairs"].append(
            {
                "src_sentence": src_sent,
                "tgt_sentence": tgt_sent,
                "src_words": src_words,
                "tgt_words": tgt_words,
                "alignments": [
                    {
                        "src_idx": src_idx,
                        "src_word": src_words[src_idx],
                        "aligned": [
                            {"tgt_idx": i, "tgt_word": tgt_words[i]}
                            for i in src_alignments[src_idx]
                        ],
                    }
                    for src_idx in src_alignments
                ],
            }
        )

    with open(test_dir / "alignments.json", "w", encoding="utf-8") as f:
        json.dump(align_json, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
