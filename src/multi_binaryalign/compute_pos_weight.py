import argparse
import random

import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from multi_binaryalign.data import BinaryAlignDataset, BinaryAlignCollator
from multi_binaryalign.tokenization import BinaryAlignTokenizer


def load_config(config_path: str) -> DictConfig:
    return OmegaConf.load(config_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute BCE pos_weight (neg/pos) over masked training tokens."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage", choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--split", choices=["train", "valid"], default="train")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--sample-with-replacement", action="store_true")
    parser.add_argument("--finetune-tgt-lang", type=str, default=None)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def build_dataset(args: argparse.Namespace, config: DictConfig) -> BinaryAlignDataset:
    manifest_path = (
        config.data.train_manifest if args.split == "train" else config.data.valid_manifest
    )

    finetune_tgt_lang = args.finetune_tgt_lang
    if finetune_tgt_lang is None:
        finetune_tgt_lang = config.data.finetune_tgt_lang

    if args.stage == "finetune" and finetune_tgt_lang is None:
        raise ValueError(
            "Finetune stage requires a target language. Set data.finetune_tgt_lang in config "
            "or pass --finetune-tgt-lang."
        )

    is_finetune = args.stage == "finetune"
    alpha = 0.0 if is_finetune else config.data.alpha

    return BinaryAlignDataset(
        manifest_path=manifest_path,
        is_finetune=is_finetune,
        finetune_tgt_lang=finetune_tgt_lang,
        alpha=alpha,
        sample_with_replacement=args.sample_with_replacement,
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    batch_size = args.batch_size or config.data.batch_size

    tokenizer = BinaryAlignTokenizer(
        model_name=config.model.backbone, max_length=config.model.max_length
    )
    collator = BinaryAlignCollator(tokenizer)
    dataset = build_dataset(args, config)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    pos_count = 0
    total_count = 0
    total_examples = 0

    iterator = enumerate(loader, start=1)
    iterator = tqdm(iterator, total=None, desc="Computing pos_weight")

    for batch_idx, batch in iterator:
        attention_mask = batch["attention_mask"].bool()
        target_mask = batch["target_mask"].bool()
        labels = batch["labels"]

        if "word_mask" in batch:
            mask = attention_mask & target_mask & batch["word_mask"].bool()
        else:
            mask = attention_mask & target_mask

        total_batch = int(mask.sum().item())
        pos_batch = int(labels[mask].sum().item())

        total_count += total_batch
        pos_count += pos_batch
        total_examples += int(labels.shape[0])

        iterator.set_postfix(
            {
                "examples": total_examples,
                "tokens": total_count,
                "pos": pos_count,
            }
        )

        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

    if total_count == 0:
        raise RuntimeError("No supervised tokens found in mask; cannot compute pos_weight.")
    if pos_count == 0:
        raise RuntimeError(
            "Found zero positive tokens in supervised mask; pos_weight would be undefined."
        )

    neg_count = total_count - pos_count
    eps = 1e-12
    pos_weight = neg_count / max(pos_count, eps)
    pos_rate = pos_count / total_count

    print("=" * 80)
    print("pos_weight statistics")
    print(f"config={args.config}")
    print(f"stage={args.stage}")
    print(f"split={args.split}")
    print(f"backbone={config.model.backbone}")
    print(f"max_length={config.model.max_length}")
    print(f"dataset_size={len(dataset)}")
    print(f"sample_with_replacement={args.sample_with_replacement}")
    print(f"batch_size={batch_size}")
    print(f"num_workers={args.num_workers}")
    if args.max_batches is not None:
        print(f"max_batches={args.max_batches}")
    print("-" * 80)
    print(f"supervised_tokens={total_count}")
    print(f"positive_tokens={pos_count}")
    print(f"negative_tokens={neg_count}")
    print(f"positive_rate={pos_rate:.8f}")
    print(f"pos_weight (neg/pos)={pos_weight:.8f}")
    print("-" * 80)
    print("Suggested loss init:")
    print(
        f"nn.BCEWithLogitsLoss(reduction='none', "
        f"pos_weight=torch.tensor({pos_weight:.8f}, device=device))"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
