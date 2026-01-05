import torch
from torch.utils.data import Dataset


class BinaryAlignDataset(Dataset):
    """


    Parameters:

    Attributes:
        src_sentences (list[list[str]]): 
        tgt_sentences (list[list[str]]): 
        alignments (list[set[tuple[int, int]]]): 

    """

    def __init__(
        self,
        src_sentences: list[list[str]],
        tgt_sentences: list[list[str]],
        alignments: list[set[tuple[int, int]]],
    ):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.alignments = alignments

        # ----------
        # Define all sentence / source word pairs
        # ----------
        self.instances = []

        for sent_idx, source_sent in enumerate(src_sentences):
            self.instances.extend([(sent_idx, src_idx) for src_idx in range(len(source_sent))])

    def __getitem__(self, idx: int):
        sent_idx, src_idx = self.instances[idx]

        src_words = self.src_sentences[sent_idx]
        tgt_words = self.tgt_sentences[sent_idx]

        tgt_word_idxs = {
            j for (i, j) in self.alignments[sent_idx]
            if i == src_idx
        }

        return {
            "src_words": src_words,
            "tgt_words": tgt_words,
            "src_word_idx": src_idx,
            "tgt_word_idxs": tgt_word_idxs
        }

    def __len__(self):
        return len(self.instances)
