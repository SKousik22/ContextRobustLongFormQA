"""
compute_bleu_scores.py

Computes BLEU scores for all JSON result files under results_qwen/ and writes 
per-query score files to bleu_scores_qwen/, mirroring the original directory structure.

Usage:
    python compute_bleu_scores.py \
        --input_dir  results_qwen \
        --output_dir bleu_scores_qwen
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Compute BLEU scores for results_qwen")
    parser.add_argument("--input_dir",   default="results_qwen")
    parser.add_argument("--output_dir",  default="bleu_scores_qwen")
    # Included to maintain CLI argument compatibility with the original script
    parser.add_argument("--device",      default=None, help="Ignored (kept for CLI compatibility)")
    parser.add_argument("--bert_model",  default="roberta-large", help="Ignored (kept for CLI compatibility)")
    parser.add_argument("--sbert_model", default="all-mpnet-base-v2", help="Ignored (kept for CLI compatibility)")
    parser.add_argument("--batch_size",  type=int, default=64, help="Ignored (kept for CLI compatibility)")
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
    """Convert to string; replace empty/whitespace-only with a single space."""
    s = str(v) if not isinstance(v, str) else v
    return s if s.strip() else " "


def is_empty(v):
    """True if the original value was empty / whitespace-only."""
    s =str(v) if not isinstance(v, str) else v
    return not s.strip()


# Score computation
def compute_bleu(references, candidates):
    try:
        nltk.data.find('tokenizers/punkt')
        use_nltk_tokenize = True
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            use_nltk_tokenize = True
        except Exception:
            use_nltk_tokenize = False

    scores = []
    chencherry = SmoothingFunction()
    
    for ref, cand in zip(references, candidates):
        if use_nltk_tokenize:
            try:
                ref_tokens = nltk.word_tokenize(ref)
                cand_tokens = nltk.word_tokenize(cand)
            except Exception:
                ref_tokens = ref.split()
                cand_tokens = cand.split()
        else:
            ref_tokens = ref.split()
            cand_tokens = cand.split()
            
        score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=chencherry.method1)
        scores.append(score)
        
    return scores


# Main
def main():
    args = parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if not input_root.exists():
        sys.exit(f"[ERROR] Input directory not found: {input_root}")

    json_files = sorted(input_root.rglob("*.json"))
    if not json_files:
        sys.exit(f"[ERROR] No .json files found under {input_root}")

    print(f"Found {len(json_files)} JSON file(s) to process.\n")

    # Process files

    empty_log = []   # collects all empty-field incidents across all files

    for json_path in tqdm(json_files, desc="Files", unit="file"):
        output_path =output_root / json_path.relative_to(input_root)

        if output_path.exists():
            tqdm.write(f" >>  Skip (exists): {output_path}")
            continue

        items =load_items(json_path)

        # Build safe ref/cand lists (no empty strings); log empty fields
        valid = []
        exp_dir  = json_path.parent.name          # e.g. "exp1a"
        filename = json_path.name                 # e.g. "0_distractors.json"
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

        # Compute BLEU
        bleu_scores = compute_bleu(references, candidates)

        # Assemble output
        output_items = []
        for (item, _, _), bleu_score in zip(valid, bleu_scores):
            output_items.append({
                "query_id":            item.get("query_id"),
                "question":            item.get("question", ""),
                "bleu":                round(bleu_score, 6),
            })

        save_scores(output_path, output_items)

    # Save empty-string log
    log_path = output_root / "empty_fields_log.json"
    output_root.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(empty_log, f, indent=4, ensure_ascii=False)
    print(f"Empty-field log ({len(empty_log)} incident(s)) → {log_path}")

    print(f"Done. Score files written to: {output_root}")


if __name__ == "__main__":
    main()