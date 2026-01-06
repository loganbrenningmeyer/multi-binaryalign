import os
import random
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from collections import defaultdict


def save_dataset(
    out_path: str,
    src_sentences: list[list[str]],
    tgt_sentences: list[list[str]],
    alignments: list[set[tuple[int, int]]]
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for src_words, tgt_words, alignment in zip(src_sentences, tgt_sentences, alignments):
            rec = {
                "src_words": src_words,
                "tgt_words": tgt_words,
                "alignment": [list(pair) for pair in sorted(alignment)]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_lines(path: str, encoding: str):
    lines = []
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            line = line.replace("\xa0", " ").strip()
            if line:
                lines.append(line)
    return lines


def read_snum_sentences(path: str) -> list[list[str]]:
    sentences = {}

    # -- Try UTF-8, if error try latin-1
    try:
        lines = read_lines(path, encoding="utf-8")
    except UnicodeDecodeError:
        lines = read_lines(path, encoding="latin-1")

    for line in lines:
        # -- Get sentence ID and text
        match = re.search(r"<s snum=(\d+)>(.*?)</s>", line, re.DOTALL)
        sent_id = int(match.group(1))
        text = match.group(2)
        # -- Store text split into words
        sentences[sent_id] = text.split()

    return sentences


def read_snum_alignments(path: str, null_id: int=None, flip: bool=False):
    alignments = defaultdict(set)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # -- Get sentence ID and alignment indices
            match = re.match(r"^\s*(\d+)\s+(\d+)\s+(\d+)(?:\s+([SP]))?\s*$", line)
            # -- Check for null ID
            if null_id is not None:
                # -- Check that src_idx and tgt_idx != null_id
                if int(match.group(2)) == null_id or int(match.group(3)) == null_id:
                    continue

            sent_id = int(match.group(1))
            src_idx = int(match.group(2)) - 1
            tgt_idx = int(match.group(3)) - 1
            conf = match.group(4)
            # -- If no sure/possible, assume sure
            if conf == "S" or conf is None:
                # -- Flip if English is target in data
                if flip:
                    alignments[sent_id].add((tgt_idx, src_idx))
                else:
                    alignments[sent_id].add((src_idx, tgt_idx))

    return alignments


def read_kftt_sentences(path: str) -> list[list[str]]:
    sentences = {}

    with open(path, "r", encoding="utf-8") as f:
        sent_id = 1
        for line in f:
            line = line.strip()
            if not line:
                continue
            sentences[sent_id] = line.split()
            sent_id += 1

    return sentences


def read_kftt_alignments(path: str):
    """
    Kyoto Free Translation Task dataset
    - Alignments originally Japanese-English, so read reversed
    for English-Japanese alignment
    """
    alignments = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        sent_id = 1
        for line in f:
            line = line.strip()
            if not line:
                continue
            # -- Reverse pairs for English-Japanese
            alignments[sent_id] = {
                tuple(map(int, reversed(pair.split("-"))))
                for pair in line.split()
            }
            sent_id += 1

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
            alignments.append([tuple(p) for p in record["alignments"]])

    return src_sentences, tgt_sentences, alignments

"""
Functions for loading sentences/alignments per-dataset to a common training format
"""
def load_snum(src_path: str, tgt_path: str, align_path: str, null_id: int=None, flip: bool=False):
    src_data = read_snum_sentences(src_path)
    tgt_data = read_snum_sentences(tgt_path)
    align_data = read_snum_alignments(align_path, null_id, flip)

    src_sentences = []
    tgt_sentences = []
    alignments = []

    for sent_id in src_data:
        src_sentences.append(src_data[sent_id])
        tgt_sentences.append(tgt_data[sent_id])
        alignments.append(align_data[sent_id])

    return src_sentences, tgt_sentences, alignments


def load_czenali(align_path: str):
    with open(align_path, encoding="utf-8") as f:
        raw = f.read()

    # Escape illegal XML chars *inside text nodes*
    raw = re.sub(
        r"<english>(.*?)</english>",
        lambda m: "<english>" + escape(m.group(1)) + "</english>",
        raw,
        flags=re.DOTALL,
    )

    raw = re.sub(
        r"<czech>(.*?)</czech>",
        lambda m: "<czech>" + escape(m.group(1)) + "</czech>",
        raw,
        flags=re.DOTALL,
    )

    root = ET.fromstring(raw)

    src_sentences = []
    tgt_sentences = []
    alignments = []

    for elem in root.iter("s"):
        english = elem.find("english").text
        czech = elem.find("czech").text
        sure = elem.find("sure").text

        if sure is None:
            continue

        src_sentences.append(english.split())
        tgt_sentences.append(czech.split())
        alignments.append({
            (int(i)-1, int(j)-1)  # 1-indexed → 0-indexed
            for i, j in (pair.split("-") for pair in sure.split())
        })

    return src_sentences, tgt_sentences, alignments


def load_kftt(src_path: str, tgt_path: str, align_path: str):
    src_data = read_kftt_sentences(src_path)
    tgt_data = read_kftt_sentences(tgt_path)
    align_data = read_kftt_alignments(align_path)

    src_sentences = []
    tgt_sentences = []
    alignments = []

    for sent_id in src_data:
        src_sentences.append(src_data[sent_id])
        tgt_sentences.append(tgt_data[sent_id])
        alignments.append(align_data[sent_id])

    return src_sentences, tgt_sentences, alignments


def split_indices(n: int, val_frac: float = 0.1, seed: int = 42):
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    n_val = int(round(n * val_frac))
    val_idx = set(idxs[:n_val])
    train_idx = [i for i in range(n) if i not in val_idx]
    val_idx = [i for i in range(n) if i in val_idx]
    return train_idx, val_idx


def main():
    root_dir = Path("./raw/")

    all_data = {}

    # ----------
    # Hansards (English-French)
    # ----------
    data_dir = root_dir / "en-fr" / "Hansards"

    src_paths = [
        data_dir / "test" / "test.e",
        data_dir / "trial" / "trial.e"
    ]
    tgt_paths = [
        data_dir / "test" / "test.f",
        data_dir / "trial" / "trial.f"
    ]
    align_paths = [
        data_dir / "test" / "test.wa.nonullalign",
        data_dir / "trial" / "trial.wa"
    ]

    hansards_data = {
        "src_lang": "en",
        "tgt_lang": "fr",
        "data": {
            "src_sentences": [],
            "tgt_sentences": [],
            "alignments": []
        }
    }
    
    for src_path, tgt_path, align_path in zip(src_paths, tgt_paths, align_paths):
        src_sentences, tgt_sentences, alignments = load_snum(src_path, tgt_path, align_path)
        hansards_data["data"]["src_sentences"].extend(src_sentences)
        hansards_data["data"]["tgt_sentences"].extend(tgt_sentences)
        hansards_data["data"]["alignments"].extend(alignments)

    all_data["hansards"] = hansards_data

    # ----------
    # Golden Collection (English-French)
    # ----------
    data_dir = root_dir / "en-fr" / "golden_collection"

    src_path = data_dir / "sentences" / "1-100-final.en"
    tgt_path = data_dir / "sentences" / "1-100-final.fr"
    align_path = data_dir / "en-fr_1-100.wa"

    fr_golden_data = {
        "src_lang": "en",
        "tgt_lang": "fr",
        "data": {
            "src_sentences": [],
            "tgt_sentences": [],
            "alignments": []
        }
    }

    src_sentences, tgt_sentences, alignments = load_snum(src_path, tgt_path, align_path)
    fr_golden_data["data"]["src_sentences"].extend(src_sentences)
    fr_golden_data["data"]["tgt_sentences"].extend(tgt_sentences)
    fr_golden_data["data"]["alignments"].extend(alignments)

    all_data["fr_golden"] = fr_golden_data

    # ----------
    # Golden Collection (English-Spanish)
    # ----------
    data_dir = root_dir / "en-es"

    src_path = data_dir / "sentences" / "1-100-final.en"
    tgt_path = data_dir / "sentences" / "1-100-final.es"
    align_path = data_dir / "en-es_1-100.wa"

    es_golden_data = {
        "src_lang": "en",
        "tgt_lang": "es",
        "data": {
            "src_sentences": [],
            "tgt_sentences": [],
            "alignments": []
        }
    }

    src_sentences, tgt_sentences, alignments = load_snum(src_path, tgt_path, align_path)

    es_golden_data["data"]["src_sentences"].extend(src_sentences)
    es_golden_data["data"]["tgt_sentences"].extend(tgt_sentences)
    es_golden_data["data"]["alignments"].extend(alignments)

    all_data["es_golden"] = es_golden_data

    # ----------
    # Golden Collection (English-Portuguese)
    # ----------
    data_dir = root_dir / "en-pt"

    src_path = data_dir / "sentences" / "1-100-final.en"
    tgt_path = data_dir / "sentences" / "1-100-final.pt"
    align_path = data_dir / "en-pt_1-100.wa"

    pt_golden_data = {
        "src_lang": "en",
        "tgt_lang": "pt",
        "data": {
            "src_sentences": [],
            "tgt_sentences": [],
            "alignments": []
        }
    }

    src_sentences, tgt_sentences, alignments = load_snum(src_path, tgt_path, align_path)

    pt_golden_data["data"]["src_sentences"].extend(src_sentences)
    pt_golden_data["data"]["tgt_sentences"].extend(tgt_sentences)
    pt_golden_data["data"]["alignments"].extend(alignments)

    all_data["pt_golden"] = pt_golden_data

    # ----------
    # CzEnAli (English-Czech)
    # ----------
    data_dir = root_dir / "en-cz" / "merged_data"
    align_paths = list(data_dir.rglob("*.wa"))

    czenali_data = {
        "src_lang": "en",
        "tgt_lang": "cz",
        "data": {
            "src_sentences": [],
            "tgt_sentences": [],
            "alignments": []
        }
    }
    
    for align_path in align_paths:
        src_sentences, tgt_sentences, alignments = load_czenali(align_path)
        czenali_data["data"]["src_sentences"].extend(src_sentences)
        czenali_data["data"]["tgt_sentences"].extend(tgt_sentences)
        czenali_data["data"]["alignments"].extend(alignments)

    all_data["czenali"] = czenali_data

    # ----------
    # ACL 2005 (English-Hindi)
    # ----------
    data_dir = root_dir / "en-hi"

    src_paths = [
        data_dir / "English-Hindi.test" / "test.e",
        data_dir / "English-Hindi.trial" / "trial.e"
    ]
    tgt_paths = [
        data_dir / "English-Hindi.test" / "test.h",
        data_dir / "English-Hindi.trial" / "trial.h"
    ]
    align_paths = [
        data_dir / "English-Hindi.test" / "test.wa.nonullalign",
        data_dir / "English-Hindi.trial" / "trial.wa"
    ]

    hindi_data = {
        "src_lang": "en",
        "tgt_lang": "hi",
        "data": {
            "src_sentences": [],
            "tgt_sentences": [],
            "alignments": []
        }
    }

    for src_path, tgt_path, align_path in zip(src_paths, tgt_paths, align_paths):
        src_sentences, tgt_sentences, alignments = load_snum(src_path, tgt_path, align_path)
        hindi_data["data"]["src_sentences"].extend(src_sentences)
        hindi_data["data"]["tgt_sentences"].extend(tgt_sentences)
        hindi_data["data"]["alignments"].extend(alignments)

    all_data["hindi"] = hindi_data

    # ----------
    # Linkoping (English-Swedish)
    # ----------
    data_dir = root_dir / "en-sv" / "ep-ensv"

    src_paths = [
        data_dir / "dev" / "dev.en.naacl",
        data_dir / "test" / "test.en.naacl"
    ]
    tgt_paths = [
        data_dir / "dev" / "dev.sv.naacl",
        data_dir / "test" / "test.sv.naacl"
    ]
    align_paths = [
        data_dir / "dev" / "dev.ensv.naacl",
        data_dir / "test" / "test.ensv.naacl"
    ]

    swedish_data = {
        "src_lang": "en",
        "tgt_lang": "sv",
        "data": {
            "src_sentences": [],
            "tgt_sentences": [],
            "alignments": []
        }
    }

    for src_path, tgt_path, align_path in zip(src_paths, tgt_paths, align_paths):
        src_sentences, tgt_sentences, alignments = load_snum(src_path, tgt_path, align_path, null_id=0)
        swedish_data["data"]["src_sentences"].extend(src_sentences)
        swedish_data["data"]["tgt_sentences"].extend(tgt_sentences)
        swedish_data["data"]["alignments"].extend(alignments)

    all_data["swedish"] = swedish_data

    # ----------
    # ACL 2005 (Romanian-English)
    # ----------
    data_dir = root_dir / "ro-en"

    dev_dir = data_dir / "dev" / "Romanian-English.test"
    test_dir = data_dir / "test" / "Romanian-English"

    dev_src_paths = sorted(list(dev_dir.rglob("*.e")))
    dev_tgt_paths = sorted(list(dev_dir.rglob("*.r")))
    dev_align_path = dev_dir / "test.wa.nonullalign"

    test_src_paths = sorted(list(test_dir.rglob("*.e")))
    test_tgt_paths = sorted(list(test_dir.rglob("*.r")))
    test_align_path = test_dir / "answers" / "test.wa.nonullalign"

    romanian_data = {
        "src_lang": "en",
        "tgt_lang": "ro",
        "data": {
            "src_sentences": [],
            "tgt_sentences": [],
            "alignments": []
        }
    }

    for src_path, tgt_path in zip(dev_src_paths, dev_tgt_paths):
        src_sentences, tgt_sentences, alignments = load_snum(src_path, tgt_path, dev_align_path, flip=True)
        romanian_data["data"]["src_sentences"].extend(src_sentences)
        romanian_data["data"]["tgt_sentences"].extend(tgt_sentences)
        romanian_data["data"]["alignments"].extend(alignments)

    for src_path, tgt_path in zip(test_src_paths, test_tgt_paths):
        src_sentences, tgt_sentences, alignments = load_snum(src_path, tgt_path, test_align_path, flip=True)
        romanian_data["data"]["src_sentences"].extend(src_sentences)
        romanian_data["data"]["tgt_sentences"].extend(tgt_sentences)
        romanian_data["data"]["alignments"].extend(alignments)

    all_data["romanian"] = romanian_data

    # ----------
    # KFTT (Japanese-English)
    # ----------
    kftt_files = [str(i).zfill(3) for i in range(1, 16)] + ["dev", "test"]
    
    src_paths = [root_dir / "ja-en" / "kftt-alignments" / "data" / f"english-{file}.txt" for file in kftt_files]
    tgt_paths = [root_dir / "ja-en" / "kftt-alignments" / "data" / f"japanese-{file}.txt" for file in kftt_files]
    align_paths = [root_dir / "ja-en" / "kftt-alignments" / "data" / f"align-{file}.txt" for file in kftt_files]

    kftt_data = {
        "src_lang": "en",
        "tgt_lang": "ja",
        "data": {
            "src_sentences": [],
            "tgt_sentences": [],
            "alignments": []
        }
    }

    for src_path, tgt_path, align_path in zip(src_paths, tgt_paths, align_paths):
        src_sentences, tgt_sentences, alignments = load_kftt(src_path, tgt_path, align_path)
        kftt_data["data"]["src_sentences"].extend(src_sentences)
        kftt_data["data"]["tgt_sentences"].extend(tgt_sentences)
        kftt_data["data"]["alignments"].extend(alignments)

    all_data["kftt"] = kftt_data
    
    # ----------
    # Create training / validation splits
    # ----------
    val_frac = 0.1
    out_dir = Path("./processed")

    train_manifest = {"data": []}
    valid_manifest = {"data": []}

    for dataset_name in all_data:
        src_lang = all_data[dataset_name]["src_lang"]
        tgt_lang = all_data[dataset_name]["tgt_lang"]

        src_sents = all_data[dataset_name]["data"]["src_sentences"]
        tgt_sents = all_data[dataset_name]["data"]["tgt_sentences"]
        alignments = all_data[dataset_name]["data"]["alignments"]

        # -- Split training / validation sentence pairs
        n = len(src_sents)
        train_idxs, valid_idxs = split_indices(n, val_frac)

        train_src = [src_sents[i] for i in train_idxs]
        train_tgt = [tgt_sents[i] for i in train_idxs]
        train_als = [alignments[i] for i in train_idxs]

        valid_src = [src_sents[i] for i in valid_idxs]
        valid_tgt = [tgt_sents[i] for i in valid_idxs]
        valid_als = [alignments[i] for i in valid_idxs]

        train_rel_path = Path(f"{src_lang}-{tgt_lang}") / f"{dataset_name}.jsonl"
        valid_rel_path = Path(f"{src_lang}-{tgt_lang}") / f"{dataset_name}.jsonl"

        save_dataset(out_dir / "train" / train_rel_path, train_src, train_tgt, train_als)
        save_dataset(out_dir / "valid" / valid_rel_path, valid_src, valid_tgt, valid_als)

        train_manifest["data"].append({
            "name": dataset_name,
            "rel_path": str(train_rel_path),
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "num_sentence_pairs": len(train_src),
            "num_alignments": sum(len(a) for a in train_als),
            "num_instances": sum(len(s) for s in train_src)
        })

        valid_manifest["data"].append({
            "name": dataset_name,
            "rel_path": str(valid_rel_path),
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "num_sentence_pairs": len(valid_src),
            "num_alignments": sum(len(a) for a in valid_als),
            "num_instances": sum(len(s) for s in valid_src)
        })

    os.makedirs(out_dir / "train", exist_ok=True)
    with open(out_dir / "train" / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(train_manifest, f, ensure_ascii=False, indent=4)

    os.makedirs(out_dir / "valid", exist_ok=True)
    with open(out_dir / "valid" / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(valid_manifest, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
