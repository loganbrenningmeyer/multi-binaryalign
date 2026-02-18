import torch

from multi_binaryalign.tokenization import BinaryAlignTokenizer
from multi_binaryalign.data.utils import get_masks, get_word_ids


class BinaryAlignCollator:
    """


    Parameters:

    """

    def __init__(self, tokenizer: BinaryAlignTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict]):
        src_batch = [x["src_words"] for x in batch]
        tgt_batch = [x["tgt_words"] for x in batch]
        src_idxs = [x["src_word_idx"] for x in batch]
        aligned_sets = [x["tgt_word_idxs"] for x in batch]

        # -------------------------
        # Encode batch / get input_ids / attention_mask
        # -------------------------
        encoding = self.tokenizer.encode_marked_batch(src_batch, tgt_batch, src_idxs)

        input_ids = encoding["input_ids"]  # (B, L)
        attention_mask = encoding["attention_mask"]  # (B, L)
        B, L = input_ids.shape

        # -------------------------
        # Get target_mask & word_mask / word_ids
        # -------------------------
        target_mask, word_mask = get_masks(encoding)
        word_ids = get_word_ids(encoding)

        # ----------
        # Initialize subword token labels
        # ----------
        labels = torch.zeros((B, L), dtype=torch.float32)

        for b in range(B):
            # -------------------------
            # Get tensor of aligned target word indices: (K_b,)
            # -- where K_b is the number of aligned target words for sample b
            # -------------------------
            aligned = torch.tensor(
                sorted(aligned_sets[b]), dtype=torch.long, device=word_ids.device
            )
            # -- Ignore no alignments
            if aligned.numel() == 0:
                continue

            # -- Checks if batch's word_ids are in aligned set (boolean)
            is_aligned = torch.isin(word_ids[b], aligned)   # (L,)

            # -- Set non-masked, aligned labels to 1
            labels[b, target_mask[b] & word_mask[b] & is_aligned] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
            "word_mask": word_mask,
            "labels": labels,
        }
