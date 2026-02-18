from transformers import AutoTokenizer, BatchEncoding


class BinaryAlignTokenizer:
    """
    
    
    Parameters:
    
    """
    def __init__(self, model_name: str, max_length: int=256, markers=["<ws>", "</ws>"]):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # ----------
        # Add source work marker tokens / resize embeddings
        # ----------
        self.tokenizer.add_special_tokens({"additional_special_tokens": markers})
        self.word_start, self.word_end = markers[0], markers[1]
        self.max_length = max_length
        self.vocab_size = len(self.tokenizer)

    def encode_marked_batch(
        self,
        src_batch: list[list[str]],
        tgt_batch: list[list[str]],
        src_idxs: list[int],
    ):
        """


        Args:


        Returns:

        """
        src_marked = self.mark_source_batch(src_batch, src_idxs)
        encoding = self.encode_batch(src_marked, tgt_batch)

        return encoding

    def encode_batch(
        self, src_batch: list[str], tgt_batch: list[str]
    ) -> BatchEncoding:
        """
        
        
        Args:
        
        
        Returns:
        
        """
        return self.tokenizer(
            src_batch,
            tgt_batch,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def mark_source_word(self, src_words: list[str], i: int) -> list[str]:
        """


        Args:


        Returns:

        """
        return (
            src_words[:i]
            + [self.word_start, src_words[i], self.word_end]
            + src_words[i + 1 :]
        )

    def mark_source_batch(
        self, src_batch: list[list[str]], src_idxs: list[int]
    ):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        assert len(src_batch) == len(src_idxs)
        return [
            self.mark_source_word(src, i)
            for src, i in zip(src_batch, src_idxs)
        ]

    def get_tokens(self, encoding: BatchEncoding, batch_idx: int):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        ids = encoding["input_ids"][batch_idx]
        tokens = self.tokenizer.convert_ids_to_tokens(ids)

        return tokens