from collections import defaultdict

from binaryalign.models import BinaryAlignModel
from binaryalign.tokenization import BinaryAlignTokenizer


class BinaryAlign:
    def __init__(
        self,
        model: BinaryAlignModel,
        tokenizer: BinaryAlignTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer

    def align_sentence_pair(
        self, src_words: list[str], tgt_words: list[str], threshold: float = 0.1
    ):
        """


        Args:


        Returns:

        """
        # -------------------------
        # Create inputs for BinaryAlignModel
        # -------------------------
        encoding, input_ids, attention_mask, target_mask = self.create_batch(
            src_words, tgt_words
        )

        # -------------------------
        # Run inference
        # -------------------------
        preds, scores, mask = self.model.predict(
            input_ids, attention_mask, target_mask, threshold
        )

        # -- B = # source words, L = # subword tokens
        B, L = preds.shape
        src_alignments = defaultdict(list)
        tgt_alignments = defaultdict(list)

        for b in range(B):
            # -- Batch b corresponds to src_words[b]
            word_idxs = encoding.word_ids(b)

            # -- Aggregate target subword scores with max
            best_score_by_tgt = {}

            # -- Iterate through all subword token indices
            for l in range(L):
                # -------------------------
                # mask: target subword token and not padding
                # preds: classified subword token as aligned
                # -------------------------
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

            # -------------------------
            # src_alignments[src_idx] = [tgt_idx_1, tgt_idx_2, ...]
            # tgt_alignments[tgt_idx] = [src_idx_1, src_idx_2, ...]
            # -------------------------
            aligned_tgt_idxs = best_score_by_tgt.keys()

            for tgt_idx in aligned_tgt_idxs:
                src_alignments[b].append(tgt_idx)
                tgt_alignments[tgt_idx].append(b)

        return src_alignments, tgt_alignments

    def align_document_pair(
        self,
        src_par_sent_words: list[list[list[str]]],
        tgt_par_sent_words: list[list[list[str]]],
        threshold: float = 0.1,
    ):
        """


        Args:


        Returns:

        """
        # -------------------------
        # Align sentence pairs / track word index offsets
        # -------------------------
        src_words_global = []
        tgt_words_global = []
        src_alignments_global = {}
        tgt_alignments_global = {}

        src_par_ids = []
        src_sent_ids = []
        src_sent_to_par_ids = {}
        src_par_to_sent_ids = defaultdict(list)

        src_par_to_word_ids = defaultdict(list)
        src_sent_to_word_ids = defaultdict(list)

        tgt_sent_ids = []
        tgt_par_ids = []

        src_offset = 0
        tgt_offset = 0

        sent_id = 0
        # -- For each paragraph...
        for par_id, (src_par, tgt_par) in enumerate(
            zip(src_par_sent_words, tgt_par_sent_words)
        ):
            # -- For words in each sentence...
            for src_words, tgt_words in zip(src_par, tgt_par):

                # -------------------------
                # Align source / target sentence pair
                # -------------------------
                src_alignments, tgt_alignments = (
                    self.align_sentence_pair(src_words, tgt_words, threshold)
                )

                # -------------------------
                # Fill global words
                # -------------------------
                src_words_global.extend(src_words)
                tgt_words_global.extend(tgt_words)

                # -------------------------
                # Update global alignments w/ src and tgt offset indices
                # -------------------------
                for src_idx, tgt_idxs in src_alignments.items():
                    tgt_idxs_global = [tgt_idx + tgt_offset for tgt_idx in tgt_idxs]
                    src_alignments_global[src_idx + src_offset] = tgt_idxs_global

                for tgt_idx, src_idxs in tgt_alignments.items():
                    src_idxs_global = [src_idx + src_offset for src_idx in src_idxs]
                    tgt_alignments_global[tgt_idx + tgt_offset] = src_idxs_global

                # -------------------------
                # Assign paragraph / sentence ids for src words
                # -------------------------
                for src_idx in range(len(src_words)):
                    src_idx_global = src_idx + src_offset
                    # -- Sentence / Paragraph IDs
                    src_sent_ids.append(sent_id)
                    src_par_ids.append(par_id)
                    # -- Sentence / Paragraph IDs --> Words
                    src_sent_to_word_ids[sent_id].append(src_idx_global)
                    src_par_to_word_ids[par_id].append(src_idx_global)
                    # -- Sentence <--> Paragraph Mappings
                    src_sent_to_par_ids[sent_id] = par_id
                    src_par_to_sent_ids[par_id].append(sent_id)

                for tgt_idx in range(len(tgt_words)):
                    tgt_sent_ids.append(sent_id)
                    tgt_par_ids.append(par_id)

                # -- Update sentence id / word index offsets
                sent_id += 1

                src_offset += len(src_words)
                tgt_offset += len(tgt_words)

        return (
            src_words_global,
            tgt_words_global,
            src_alignments_global,
            tgt_alignments_global,
            src_sent_ids,
            src_sent_to_par_ids,
            src_sent_to_word_ids,
            src_par_ids,
            src_par_to_sent_ids,
            src_par_to_word_ids,
            tgt_sent_ids,
            tgt_par_ids,
        )

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
