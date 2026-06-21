"""
compute_scores_qwen.py

Computes BERTScore, ROUGE-L, and Sentence-BERTScore for all JSON result files
under results_qwen/ and writes per-query score files to base_scores_qwen/,
mirroring the original directory structure.

Usage:
    python compute_scores_qwen.py \
        --input_dir  results_qwen \
        --output_dir base_scores_qwen \
        [--device cuda] \
        [--bert_model roberta-large] \
        [--sbert_model all-mpnet-base-v2] \
        [--batch_size 64]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import logging as hf_logging
from rouge_score import rouge_scorer as rs_mod
from bert_score import BERTScorer

# Silence HuggingFace init warnings globally
hf_logging.set_verbosity_error()
# Silence any remaining Python-level warnings from bert_score
warnings.filterwarnings("ignore", message=".*weights.*not initialized.*")
warnings.filterwarnings("ignore", message=".*Empty.*sentence.*")


# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Compute NLG scores for results_qwen")
    parser.add_argument("--input_dir",   default="results_qwen")
    parser.add_argument("--output_dir",  default="base_scores_qwen")
    parser.add_argument("--device",      default=None,
                        help="'cuda' or 'cpu'. Auto-detected if omitted.")
    parser.add_argument("--bert_model",  default="roberta-large")
    parser.add_argument("--sbert_model", default="all-mpnet-base-v2")
    parser.add_argument("--batch_size",  type=int, default=64)
    return parser.parse_args()


# Helpers
def load_items(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_scores(path: Path, scores: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4, ensure_ascii=False)


def safe_str(v):
    """Convert to string; replace empty/whitespace-only with a single space
    so BERTScore never sees an empty sentence."""
    s=str(v) if not isinstance(v, str) else v
    return s if s.strip() else " "


def is_empty(v):
    """True if the original value was empty / whitespace-only."""
    s=str(v) if not isinstance(v, str) else v
    return not s.strip()


# Score computation
def compute_rougeL(references, candidates):
    scorer = rs_mod.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(r, c)["rougeL"].fmeasure
            for r, c in zip(references, candidates)]


# Main
def main():
    args = parse_args()

    device =args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_root  =Path(args.input_dir)
    output_root =Path(args.output_dir)

    if not input_root.exists():
        sys.exit(f"[ERROR] Input directory not found: {input_root}")

    json_files = sorted(input_root.rglob("*.json"))
    if not json_files:
        sys.exit(f"[ERROR] No .json files found under {input_root}")

    print(f"Found {len(json_files)} JSON file(s) to process.\n")

    #Load models
    # BERTScore
    print(f"Loading BERTScore model: {args.bert_model} ...")
    bert_scorer = BERTScorer(
        model_type=args.bert_model,
        device=device,
        batch_size=args.batch_size,
        lang="en",
        rescale_with_baseline=False,
    )
    print("BERTScore model loaded.")

    # Sentence-BERT
    print(f"Loading Sentence-BERT model: {args.sbert_model} ...")
    from sentence_transformers import SentenceTransformer
    import torch.nn.functional as F
    sbert_model = SentenceTransformer(args.sbert_model, device=device)
    print("Sentence-BERT model loaded.\n")

    # Process files
    empty_log = []   # collects all empty-field incidents across all files

    for json_path in tqdm(json_files, desc="Files", unit="file"):
        output_path = output_root / json_path.relative_to(input_root)

        if output_path.exists():
            tqdm.write(f"  >>  Skip (exists): {output_path}")
            continue

        items = load_items(json_path)

        # Build safe ref/cand lists (no empty strings); log empty fields
        valid = []
        exp_dir  = json_path.parent.name          
        filename = json_path.name                 
        for item in items:
            raw_ref  = item.get("reference_answer",  "")
            raw_cand = item.get("generated_answer", "")
            empty_fields = []
            if is_empty(raw_ref):
                empty_fields.append("reference_answer")
            if is_empty(raw_cand):
                empty_fields.append("generated_answer")
            if empty_fields:
                empty_log.append({
                    "exp_dir":      exp_dir,
                    "file":         filename,
                    "query_id":     item.get("query_id"),
                    "empty_fields": empty_fields,
                })
            ref  = safe_str(raw_ref)
            cand = safe_str(raw_cand)
            valid.append((item, ref, cand))

        if not valid:
            tqdm.write(f"  [SKIP] No valid items in {json_path}")
            continue

        references = [v[1] for v in valid]
        candidates = [v[2] for v in valid]

        # BERTScore — single model call, no re-init
        _, _, F1 = bert_scorer.score(candidates, references)
        bert_f1s = F1.tolist()

        # ROUGE-L
        rougeL_scores = compute_rougeL(references, candidates)

        # Sentence-BERTScore
        ref_embs  = sbert_model.encode(references,  batch_size=args.batch_size,
                                        convert_to_tensor=True, show_progress_bar=False)
        cand_embs = sbert_model.encode(candidates,  batch_size=args.batch_size,
                                        convert_to_tensor=True, show_progress_bar=False)
        sbert_scores = F.cosine_similarity(ref_embs, cand_embs, dim=1).tolist()

        # Assemble output
        output_items = []
        for (item, _, _), bf1, rl, sb in zip(valid, bert_f1s, rougeL_scores, sbert_scores):
            output_items.append({
                "query_id":            item.get("query_id"),
                "question":            item.get("question", ""),
                "bert-score":          round(bf1, 6),
                "sentence-bert-score": round(sb,  6),
                "rouge-l":             round(rl,  6),
            })

        save_scores(output_path, output_items)

    # Save empty-string log
    log_path = output_root / "empty_fields_log.json"
    output_root.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(empty_log, f, indent=4, ensure_ascii=False)
    print(f"Empty-field log ({len(empty_log)} incident(s)) --> {log_path}")

    print(f"Done. Score files written to: {output_root}")


if __name__ == "__main__":
    main()
