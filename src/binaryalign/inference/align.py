import torch
import torch.nn as nn
import spacy
from transformers import AutoModel
from collections import defaultdict

from binaryalign.models import BinaryAlignModel
from binaryalign.tokenization import BinaryAlignTokenizer


class BinaryAlign:
    def __init__(
            self, 
            model: BinaryAlignModel, 
            tokenizer: BinaryAlignTokenizer, 
            src_lang: str, 
            tgt_lang: str
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_nlp = spacy.blank(src_lang)
        self.tgt_nlp = spacy.blank(tgt_lang)


    def align(self, src_sentence: str, tgt_sentence: str, threshold: float):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        # ----------
        # Break sentences into words 
        # ----------
        src_words, tgt_words = self.tokenize_sentences(src_sentence, tgt_sentence)

        # ----------
        # Create inputs for BinaryAlignModel
        # ----------
        encoding, input_ids, attention_mask, target_mask = self.create_batch(src_words, tgt_words)

        # ----------
        # Run inference
        # ----------
        preds, scores, mask = self.model.predict(input_ids, attention_mask, target_mask, threshold)

        # -- B = # source words, L = # subword tokens
        B, L = preds.shape
        alignments = defaultdict(list)

        for b in range(B):
            # -- Batch b corresponds to src_words[b]
            src_word = src_words[b]
            word_idxs = encoding.word_ids(b)

            # -- Aggregate target subword scores with max
            best_score_by_tgt = {}

            # -- Iterate through all subword token indices
            for l in range(L):
                # ----------
                # mask: target subword token and not padding
                # preds: classified subword token as aligned
                # ----------
                if mask[b, l] and preds[b, l]:
                    # -- subword token l --> target word index
                    tgt_word_idx = word_idxs[l]
                    # -- Ignore special tokens
                    if tgt_word_idx is None:
                        continue
                    # -- Logit for batch b subword token l
                    score = float(scores[b, l].item())
                    # -- Track target words' best score (max aggregation)
                    prev = best_score_by_tgt.get(tgt_word_idx, -1.0)
                    if score > prev:
                        best_score_by_tgt[tgt_word_idx] = score

            # ----------
            # alignments dictionary stores the aligned target words,
            # their positions, and scores for each source word:
            # 
            #   alignments[src_word_idx, src_word] = [
            #       (tgt_word_idx, tgt_word, tgt_score),
            #       ...
            #   ]
            # ----------
            alignments[(b, src_word)] = [
                (i, tgt_words[i], best_score_by_tgt[i])
                for i in sorted(best_score_by_tgt.keys())
            ]

        return src_words, tgt_words, alignments
    
    def tokenize_sentences(self, src_sentence: str, tgt_sentence: str):
        # -- Break sentences into words
        src_words = [t.text for t in self.src_nlp(src_sentence)]
        tgt_words = [t.text for t in self.tgt_nlp(tgt_sentence)]
        return src_words, tgt_words

    def create_batch(self, src_words: list[str], tgt_words: list[str]):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        # -- Align for all source words
        src_idxs = list(range(len(src_words)))

        # -- Form batch for each source word
        src_batch = [src_words] * len(src_idxs)
        tgt_batch = [tgt_words] * len(src_idxs)

        # -- Mark / encode batch
        encoding = self.tokenizer.encode_marked_batch(src_batch, tgt_batch, src_idxs)

        input_ids = encoding["input_ids"].to(self.model.device)
        attention_mask = encoding["attention_mask"].to(self.model.device)
        target_mask = (encoding["token_type_ids"] == 1).to(self.model.device)

        return encoding, input_ids, attention_mask, target_mask


