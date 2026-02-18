import json
import random
from pathlib import Path
from torch.utils.data import Dataset
from collections import defaultdict


class BinaryAlignDataset(Dataset):
    """
    
    
    Parameters:
    
    """
    def __init__(
        self, 
        manifest_path: str, 
        is_finetune: bool,
        finetune_tgt_lang: str=None, 
        alpha: float=0.5,
        sample_with_replacement: bool=True
    ):
        if is_finetune and finetune_tgt_lang is None:
            raise ValueError("When is_finetune, finetune_tgt_lang must be specified.")
        
        # ----------
        # Read datasets manifest information
        # ----------
        with open(manifest_path, "r") as f:
            manifest = json.load(f)["data"]

        # ----------
        # Load datasets for specified target languages
        # ----------
        self.datasets = []
        self.weights = []

        for entry in manifest:
            # -------------------------
            # Finetuning: Use only finetune target language
            # -------------------------
            if is_finetune and entry["tgt_lang"] != finetune_tgt_lang:
                continue
            # -------------------------
            # Pretraining:
            # - finetune_tgt_lang == None: Use all languages
            # - finetune_tgt_lang != None: Use all non-finetune languages
            # -------------------------
            elif not is_finetune and finetune_tgt_lang != None:
                # -- All non-finetune languages
                if entry["tgt_lang"] == finetune_tgt_lang:
                    continue
            else:
                # -- Dataset json path relative to manifest location
                jsonl_path = Path(manifest_path).parent / entry["rel_path"]

                ds = LanguagePairDataset(jsonl_path)
                self.datasets.append(ds)

                # -- Temperature-weighted size
                weight = len(ds) ** alpha
                self.weights.append(weight)

        # ----------
        # Deterministic sampling for validation
        # ----------
        self.sample_with_replacement = sample_with_replacement

        if not sample_with_replacement:
            self.flat = []
            for di, ds in enumerate(self.datasets):
                for j in range(len(ds)):
                    self.flat.append((di, j))

    def __getitem__(self, idx):
        # ----------
        # Random weighted sampling (training)
        # ----------
        if self.sample_with_replacement:
            # -- Select random language pair dataset
            ds = random.choices(self.datasets, weights=self.weights, k=1)[0]
            # -- Sample random instance from chosen dataset
            j = random.randrange(len(ds))
            return ds[j]
        # ----------
        # Deterministic (validation)
        # ----------
        else:
            di, j = self.flat[idx]
            return self.datasets[di][j]

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)


class LanguagePairDataset(Dataset):
    """

    """

    def __init__(self, jsonl_path: str):
        # ----------
        # Read language pair jsonl data
        # ----------
        data = self._parse_jsonl(jsonl_path)

        self.src_sentences = data["src_sentences"]
        self.tgt_sentences = data["tgt_sentences"]
        self.alignments = data["alignments"]

        # ----------
        # Define all sentence / source word pairs
        # ----------
        self.instances = []

        for sent_idx, source_sent in enumerate(self.src_sentences):
            self.instances.extend([(sent_idx, src_idx) for src_idx in range(len(source_sent))])

        # ----------
        # Define src_idx -> tgt_idxs mapping
        # ----------
        self.aligned = []
        for sent_pairs in self.alignments:
            pairs = defaultdict(set)
            for src_word_idx, tgt_word_idx in sent_pairs:
                pairs[src_word_idx].add(tgt_word_idx)
            self.aligned.append(pairs)

    def __getitem__(self, idx: int):
        sent_idx, src_idx = self.instances[idx]

        src_words = self.src_sentences[sent_idx]
        tgt_words = self.tgt_sentences[sent_idx]

        tgt_word_idxs = self.aligned[sent_idx].get(src_idx, set())

        return {
            "src_words": src_words,
            "tgt_words": tgt_words,
            "src_word_idx": src_idx,
            "tgt_word_idxs": tgt_word_idxs
        }

    def __len__(self):
        return len(self.instances)

    def _parse_jsonl(self, jsonl_path: str):
        data = {
            "src_sentences": [],
            "tgt_sentences": [],
            "alignments": []
        }

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line_data = json.loads(line)

                data["src_sentences"].append(line_data["src_words"])
                data["tgt_sentences"].append(line_data["tgt_words"])
                data["alignments"].append(line_data["alignment"])

        return data