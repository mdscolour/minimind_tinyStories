from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

# =========================================================
# Paths (relative, cross-platform)
# =========================================================
repo_dir = Path(__file__).resolve().parent
dataset_dir = repo_dir / "dataset"
dataset_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# Config
# =========================================================
TMP_PRETRAIN_N = 15000        # for SFT and DPO
FINAL_PRETRAIN_N = 10000      # for pretrain
N_SFT = 200
N_DPO = 200

TMP_PRETRAIN_PATH = dataset_dir / f"pretrain_hq_tmp_{TMP_PRETRAIN_N}.jsonl"
FINAL_PRETRAIN_PATH = dataset_dir / f"pretrain_hq_{FINAL_PRETRAIN_N}.jsonl"

OUT_SFT = dataset_dir / f"sft_tinystories_{N_SFT}.jsonl"
OUT_DPO = dataset_dir / f"dpo_tinystories_{N_DPO}.jsonl"

RNG_SEED = 42
random.seed(RNG_SEED)

# =========================================================
# Helpers for SFT/DPO building (from your first script)
# =========================================================
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"\s+")

def normalize_ws(s: str) -> str:
    return _WS.sub(" ", s).strip()

def extract_text(record: Dict[str, Any]) -> Optional[str]:
    for k in ("text", "content", "story", "completion"):
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            return v
    for v in record.values():
        if isinstance(v, str) and v.strip():
            return v
    return None

def split_first_sentence(text: str) -> Optional[Tuple[str, str]]:
    text = normalize_ws(text)
    if not text:
        return None

    sentences = _SENT_SPLIT.split(text)
    sentences = [normalize_ws(s) for s in sentences if normalize_ws(s)]

    if len(sentences) < 2:
        return None

    first = sentences[0]
    rest = " ".join(sentences[1:]).strip()
    if not rest:
        return None
    return first, rest

def make_prompt(first_sentence: str) -> str:
    return f"Continue the story: {first_sentence}"

def make_rejected(rest_text: str) -> str:
    rest_text = normalize_ws(rest_text)
    sentences = _SENT_SPLIT.split(rest_text)
    sentences = [normalize_ws(s) for s in sentences if normalize_ws(s)]

    if len(sentences) < 2:
        s = sentences[0] if sentences else rest_text
        return normalize_ws((s + " ") * 6)

    mode = random.choice(["shuffle", "repeat", "inject"])
    if mode == "shuffle":
        shuffled = sentences[:]
        random.shuffle(shuffled)
        return " ".join(shuffled)

    if mode == "repeat":
        s = random.choice(sentences)
        return normalize_ws((s + " ") * 6)

    injection = random.choice([
        "As a financial advisor, I will now explain the tax implications in detail.",
        "Here is a technical explanation with complex terms that kids do not need.",
        "This is an unrelated lecture about statistics and optimization methods.",
    ])
    tail = " ".join(sentences[: max(1, len(sentences)//2)])
    return normalize_ws(injection + " " + tail)

def read_last_n_jsonl(path: Path, n: int) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [ln for ln in lines if ln.strip()]
    tail = lines[-n:] if n <= len(lines) else lines

    out: List[Dict[str, Any]] = []
    for ln in tail:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out

def build_sft(records: List[Dict[str, Any]], target_n: int) -> List[Dict[str, Any]]:
    sft: List[Dict[str, Any]] = []
    for r in records:
        if len(sft) >= target_n:
            break
        txt = extract_text(r)
        if not txt:
            continue
        split = split_first_sentence(txt)
        if not split:
            continue
        first, rest = split
        sft.append({
            "conversations": [
                {"role": "user", "content": make_prompt(first)},
                {"role": "assistant", "content": rest},
            ]
        })
    return sft

def build_dpo(records: List[Dict[str, Any]], target_n: int) -> List[Dict[str, Any]]:
    dpo: List[Dict[str, Any]] = []
    for r in records:
        if len(dpo) >= target_n:
            break
        txt = extract_text(r)
        if not txt:
            continue
        split = split_first_sentence(txt)
        if not split:
            continue
        first, rest = split
        prompt = make_prompt(first)
        chosen = rest
        rejected = make_rejected(rest)

        dpo.append({
            "chosen": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen},
            ],
            "rejected": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected},
            ],
        })
    return dpo

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# =========================================================
# TinyStories → jsonl writer
# =========================================================
def write_tinystories_jsonl(out_path: Path, n_rows: int) -> int:
    """
    Download TinyStories and write first n_rows to out_path.
    Note: dataset is big; this streams iteration, no need to hold all in memory.
    """
    print("Downloading TinyStories (streaming through dataset)...")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    written = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in tqdm(ds, desc=f"Writing {out_path.name}", total=None):
            f.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")
            written += 1
            if written >= n_rows:
                break

    print(f"✔ {out_path.name} written: {written}")
    return written

# =========================================================
# Main pipeline
# =========================================================
def main() -> None:
    print("repo_dir   =", repo_dir)
    print("dataset_dir=", dataset_dir)

    # 1) Temporary pretrain：15000
    write_tinystories_jsonl(TMP_PRETRAIN_PATH, TMP_PRETRAIN_N)

    # 2) Real dataset for pretrain：10000
    write_tinystories_jsonl(FINAL_PRETRAIN_PATH, FINAL_PRETRAIN_N)

    # 3) SFT/DPO
    if not TMP_PRETRAIN_PATH.exists():
        raise FileNotFoundError(f"Temp pretrain not found: {TMP_PRETRAIN_PATH}")

    tail_n = max(N_SFT, N_DPO) * 3  # cushion for skips
    tail_records = read_last_n_jsonl(TMP_PRETRAIN_PATH, tail_n)

    sft_rows = build_sft(tail_records, N_SFT)
    dpo_rows = build_dpo(tail_records, N_DPO)

    if len(sft_rows) < N_SFT:
        print(f"[WARN] Only built {len(sft_rows)} SFT rows (requested {N_SFT}).")
    if len(dpo_rows) < N_DPO:
        print(f"[WARN] Only built {len(dpo_rows)} DPO rows (requested {N_DPO}).")

    write_jsonl(OUT_SFT, sft_rows)
    write_jsonl(OUT_DPO, dpo_rows)

    print("✔ Wrote:")
    print("  ", TMP_PRETRAIN_PATH)
    print("  ", FINAL_PRETRAIN_PATH)
    print("  ", OUT_SFT)
    print("  ", OUT_DPO)
    print(f"Seed used for rejected generation: {RNG_SEED}")

if __name__ == "__main__":
    main()