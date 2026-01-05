import os
import json
import re
from collections import defaultdict


def read_sentences(path: str) -> list[list[str]]:
    sentences = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # -- Get sentence ID and text
            match = re.search(r"<s snum=(\d+)>(.*?)</s>", line, re.DOTALL)
            sent_id = int(match.group(1))
            text = match.group(2)
            # -- Store text split into words
            sentences[sent_id] = text.split()

    return sentences


def read_alignments(path: str):
    alignments = defaultdict(set)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # -- Get sentence ID and alignment indices
            match = re.search(r"(\d+) (\d+) (\d+) .", line, re.DOTALL)
            sent_id = int(match.group(1))
            src_idx = int(match.group(2)) - 1
            tgt_idx = int(match.group(3)) - 1

            alignments[sent_id].add((src_idx, tgt_idx))

    return alignments


def save_alignment_jsonl(path, src_sentences, tgt_sentences, alignments):
    with open(path, "w", encoding="utf-8") as f:
        for src, tgt, al in zip(src_sentences, tgt_sentences, alignments):
            record = {
                "src_words": src,
                "tgt_words": tgt,
                "alignments": [list(pair) for pair in al],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_alignment_jsonl(path):
    src_sentences = []
    tgt_sentences = []
    alignments = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            src_sentences.append(record["src_words"])
            tgt_sentences.append(record["tgt_words"])
            alignments.append(set(tuple(p) for p in record["alignments"]))

    return src_sentences, tgt_sentences, alignments


def load_hansards(src_path: str, tgt_path: str, align_path: str):
    src_data = read_sentences(src_path)
    tgt_data = read_sentences(tgt_path)
    align_data = read_alignments(align_path)

    src_sentences = []
    tgt_sentences = []
    alignments = []

    for sent_id in src_data:
        src_sentences.append(src_data[sent_id])
        tgt_sentences.append(tgt_data[sent_id])
        alignments.append(align_data[sent_id])

    return src_sentences, tgt_sentences, alignments
