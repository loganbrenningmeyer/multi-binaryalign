import argparse

import torch
from omegaconf import OmegaConf, DictConfig

from multi_binaryalign.data import BinaryAlignDataset, BinaryAlignCollator
from multi_binaryalign.tokenization import BinaryAlignTokenizer


def load_config(config_path: str) -> DictConfig:
    return OmegaConf.load(config_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one collated batch and print attention/target masks per token."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", choices=["train", "valid"], default="train")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-idx", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-token-rows", type=int, default=None)
    parser.add_argument("--is-finetune", action="store_true")
    parser.add_argument("--finetune-tgt-lang", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--sample-with-replacement", action="store_true")
    return parser.parse_args()


def _word_for_token(
    token_type_id: int,
    word_id: int | None,
    src_marked_words: list[str],
    tgt_words: list[str],
) -> str:
    if word_id is None:
        return "-"
    if token_type_id == 0:
        if word_id < len(src_marked_words):
            return src_marked_words[word_id]
        return "<src_oob>"
    if word_id < len(tgt_words):
        return tgt_words[word_id]
    return "<tgt_oob>"


def _sample_batch(dataset: BinaryAlignDataset, start: int, end: int) -> list[dict]:
    return [dataset[i] for i in range(start, end)]


def main():
    args = parse_args()
    config = load_config(args.config)

    batch_size = args.batch_size or config.data.batch_size
    alpha = config.data.alpha if args.alpha is None else args.alpha

    finetune_tgt_lang = args.finetune_tgt_lang
    if finetune_tgt_lang is None:
        finetune_tgt_lang = config.data.finetune_tgt_lang

    manifest_path = (
        config.data.train_manifest if args.split == "train" else config.data.valid_manifest
    )

    tokenizer = BinaryAlignTokenizer(
        model_name=config.model.backbone,
        max_length=config.model.max_length,
    )
    collator = BinaryAlignCollator(tokenizer)

    dataset = BinaryAlignDataset(
        manifest_path=manifest_path,
        is_finetune=args.is_finetune,
        finetune_tgt_lang=finetune_tgt_lang,
        alpha=alpha,
        sample_with_replacement=args.sample_with_replacement,
    )

    start_idx = args.batch_idx * batch_size
    if start_idx >= len(dataset):
        raise ValueError(
            f"batch_idx {args.batch_idx} is out of range for dataset size {len(dataset)} "
            f"with batch_size {batch_size}."
        )

    end_idx = min(start_idx + batch_size, len(dataset))
    batch_samples = _sample_batch(dataset, start_idx, end_idx)

    if args.num_samples is not None:
        if args.num_samples < 1:
            raise ValueError("num_samples must be >= 1 when provided.")
        batch_samples = batch_samples[: args.num_samples]

    batch = collator(batch_samples)

    src_batch = [x["src_words"] for x in batch_samples]
    tgt_batch = [x["tgt_words"] for x in batch_samples]
    src_idxs = [x["src_word_idx"] for x in batch_samples]
    encoding = tokenizer.encode_marked_batch(src_batch, tgt_batch, src_idxs)

    print("=" * 100)
    print("Batch Inspection")
    print(f"split={args.split}")
    print(f"manifest={manifest_path}")
    print(f"model.backbone={config.model.backbone}")
    print(f"model.max_length={config.model.max_length}")
    print(f"dataset_size={len(dataset)}")
    print(f"batch_idx={args.batch_idx}, batch_size={batch_size}, sampled={len(batch_samples)}")
    print(f"sample_with_replacement={args.sample_with_replacement}")
    print("=" * 100)

    if not torch.equal(batch["input_ids"], encoding["input_ids"]):
        print("Warning: collator input_ids do not match freshly re-encoded input_ids.")

    for b, sample in enumerate(batch_samples):
        src_words = sample["src_words"]
        tgt_words = sample["tgt_words"]
        src_word_idx = sample["src_word_idx"]
        aligned_tgt_idxs = sorted(sample["tgt_word_idxs"])

        src_marked = tokenizer.mark_source_word(src_words, src_word_idx)
        tokens = tokenizer.get_tokens(encoding, b)
        word_ids = encoding.word_ids(b)
        token_type_ids = encoding["token_type_ids"][b].tolist()

        attention_mask = batch["attention_mask"][b].tolist()
        target_mask = batch["target_mask"][b].tolist()
        labels = batch["labels"][b].tolist()

        loss_mask = batch["attention_mask"][b].bool() & batch["target_mask"][b].bool()
        n_attend = int(batch["attention_mask"][b].sum().item())
        n_target = int(batch["target_mask"][b].sum().item())
        n_loss = int(loss_mask.sum().item())
        n_positive = int(batch["labels"][b][loss_mask].sum().item())

        print()
        print("-" * 100)
        print(f"sample[{b}]")
        print(f"src_word_idx={src_word_idx}, src_word={src_words[src_word_idx]}")
        print(f"aligned_tgt_idxs={aligned_tgt_idxs}")
        print(
            "aligned_tgt_words="
            + str([tgt_words[i] for i in aligned_tgt_idxs if i < len(tgt_words)])
        )
        print(f"src_words={src_words}")
        print(f"src_marked_words={src_marked}")
        print(f"tgt_words={tgt_words}")
        print(
            f"counts: attended={n_attend}, target_mask=1={n_target}, "
            f"loss_mask(attention&target)={n_loss}, positives_in_loss_mask={n_positive}"
        )
        print("-" * 100)
        print(
            "tok_idx token                    attn tgt_mask label token_type side word_id word"
        )
        print("-" * 100)

        rows = len(tokens)
        if args.max_token_rows is not None:
            rows = min(rows, args.max_token_rows)

        for i in range(rows):
            token = tokens[i].replace("\n", "\\n")
            word_id = word_ids[i]
            token_type = token_type_ids[i]
            attn = int(attention_mask[i])
            tgt_m = int(target_mask[i])
            label = int(labels[i])

            if attn == 0:
                side = "pad"
            elif token_type == 0:
                side = "src"
            else:
                side = "tgt"

            word = _word_for_token(token_type, word_id, src_marked, tgt_words)
            word_id_str = "-" if word_id is None else str(word_id)

            print(
                f"{i:>7} {token:<24} {attn:>4} {tgt_m:>8} {label:>5} "
                f"{token_type:>10} {side:<4} {word_id_str:>7} {word}"
            )

        if rows < len(tokens):
            print(f"... truncated {len(tokens) - rows} rows (use --max-token-rows to increase).")


if __name__ == "__main__":
    main()
