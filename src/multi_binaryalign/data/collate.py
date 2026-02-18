import torch

from multi_binaryalign.tokenization import BinaryAlignTokenizer


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

        # ----------
        # Initialize target_mask / word_mask / labels
        # ----------
        target_mask = torch.zeros((B, L), dtype=torch.bool)
        word_mask = torch.zeros((B, L), dtype=torch.bool)
        labels = torch.zeros((B, L), dtype=torch.float32)

        for b in range(B):
            seq_ids = encoding.sequence_ids(b)  # 0=src, 1=tgt, None=special/pad
            word_ids = encoding.word_ids(b) # token -> word index
            aligned_set = aligned_sets[b]  # {aligned target word indices}

            word_idxs = encoding.word_ids(b)  # token -> word index

            for l, (seq_id, word_id) in enumerate(zip(seq_ids, word_ids)):
                # -------------------------
                # Determine target_mask / word_mask
                # -------------------------
                is_tgt = seq_id == 1
                is_word = word_id is not None

                target_mask[b, l] = is_tgt
                word_mask[b, l] = is_word

                # -------------------------
                # Set alignment label if is an aligned target word
                # -------------------------
                if is_tgt and is_word and (word_id in aligned_set):
                    labels[b, l] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
            "word_mask": word_mask,
            "labels": labels,
        }
