import torch
from transformers import BatchEncoding


def get_word_ids(encoding: BatchEncoding) -> torch.Tensor:
    """
    Constructs (B, L) word id tensor from BatchEncoding.

    Non-word positions (special tokens / padding) are set to -1.
    """
    B, L = encoding["input_ids"].shape
    word_ids = torch.full((B, L), fill_value=-1, dtype=torch.long)

    for b in range(B):
        # -- Batch word_ids: list length L
        word_ids_b = encoding.word_ids(b)
        # -- For each subword token l in batch...
        for l, word_id in enumerate(word_ids_b):
            if word_id is not None:
                word_ids[b, l] = word_id

    return word_ids


def get_masks(encoding: BatchEncoding) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs target_mask and word_mask from BatchEncoding
    """
    B, L = encoding["input_ids"].shape

    # ----------
    # Initialize target_mask / word_mask / labels
    # ----------
    target_mask = torch.zeros((B, L), dtype=torch.bool)
    word_mask = torch.zeros((B, L), dtype=torch.bool)

    for b in range(B):
        seq_ids = encoding.sequence_ids(b) # 0=src, 1=tgt, None=special/pad
        word_ids = encoding.word_ids(b) # token -> word index

        for l, (seq_id, word_id) in enumerate(zip(seq_ids, word_ids)):
            # -------------------------
            # Determine target_mask / word_mask for subword token
            # -------------------------
            is_tgt = seq_id == 1
            is_word = word_id is not None

            target_mask[b, l] = is_tgt
            word_mask[b, l] = is_word

    return target_mask, word_mask
