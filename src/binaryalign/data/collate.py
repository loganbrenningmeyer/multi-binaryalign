import torch

from binaryalign.tokenization import BinaryAlignTokenizer


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

        # ----------
        # Encode marked source words / get target mask
        # ----------
        encoding = self.tokenizer.encode_marked_batch(src_batch, tgt_batch, src_idxs)
        tgt_mask = self.tokenizer.get_target_mask_batch(encoding)

        input_ids = encoding["input_ids"] # (B, L)
        attention_mask = encoding["attention_mask"] # (B, L)
        B, L = input_ids.shape

        # ----------
        # Create alignment labels
        # ----------
        labels = torch.zeros((B, L), dtype=torch.float32)

        for b in range(B):
            aligned_set = aligned_sets[b] # {aligned target word indices}
            word_idxs = encoding.word_ids(b) # token -> word index
            seq_ids = encoding.sequence_ids(b) # token -> source (0) or target (1)

            for l, seq_id in enumerate(seq_ids):
                # ----------
                # If target word is aligned, label = 1
                # ----------
                if seq_id == 1 and word_idxs[l] in aligned_set:
                    labels[b, l] = 1.0

        labels_target = labels[tgt_mask]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": tgt_mask,
            "labels_target": labels_target
        }


        