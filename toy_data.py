from pathlib import Path
import json

# repo root = this file's parent directory
repo_dir = Path(__file__).resolve().parent
dataset_dir = repo_dir / "dataset"
dataset_dir.mkdir(parents=True, exist_ok=True)

print("Writing dataset to:", dataset_dir)

# PRETRAIN
pretrain = [
    {"text": "Language models predict the next token."},
    {"text": "Society shapes behavior through norms and institutions."}
]
with open(dataset_dir/"pretrain_hq_test.jsonl","w") as f:
    for r in pretrain:
        f.write(json.dumps(r)+"\n")

# SFT
sft = [
    {"conversations":[
        {"role":"user","content":"What is AI?"},
        {"role":"assistant","content":"AI is a system that learns patterns from data."}
    ]}
]
with open(dataset_dir/"sft_mini_test.jsonl","w") as f:
    for r in sft:
        f.write(json.dumps(r)+"\n")

# DPO
dpo = [
    {
        "chosen":[
            {"role":"user","content":"Explain politely"},
            {"role":"assistant","content":"Sure, let me explain clearly."}
        ],
        "rejected":[
            {"role":"user","content":"Explain politely"},
            {"role":"assistant","content":"Figure it out yourself."}
        ]
    }
]
with open(dataset_dir/"dpo_test.jsonl","w") as f:
    for r in dpo:
        f.write(json.dumps(r)+"\n")

print("Toy dataset written.")
