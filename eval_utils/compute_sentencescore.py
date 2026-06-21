"""
PropScore computation as defined in:
  "Proposition-Level Evaluation with Adaptive Top-p DCG Aggregation"

Pipeline:
  1. Embed all propositions with BAAI/bge-large-en-v1.5  (bi-encoder)
  2. Compute cosine similarity → AdjSim
  3. Score every (r, g) pair with cross-encoder/nli-deberta-v3-large → Pf, Pb, Psym
  4. Compute PS(r, g) for gamma ∈ [1 .. 15]
  5. For each r_ij sort PS over k, take top-p = ceil(sqrt(m_i)) matches,
     compute AggPS via DCG, then average --> PropScore(i)

Usage:
  python compute_propscore.py \
      --input_dir  results_qwen \
      --output_dir sentence_propscore_qwen \
      [--embed_batch_size 256] [--nli_batch_size 64]

"""

import argparse
import json
import math
import os
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


 # NLTK punkt
import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# Constants
GAMMAS      = list(range(1, 16))          
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
NLI_MODEL   = "cross-encoder/nli-deberta-v3-large"

ENTAILMENT_IDX = None   # resolved at runtime from model's id2label, if you need fixed index, set it here (e.g. 0 for MNLI-based models)


# Model helpers
def resolve_entailment_idx(model) -> int:
    id2label = model.config.id2label
    for idx, label in id2label.items():
        if label.lower() == "entailment":
            return int(idx)
    raise ValueError(f"'entailment' not found in id2label: {id2label}")


# Sentence splitting
def split_sentences(text: str) -> list:
    sents = sent_tokenize(text.strip())
    return [s.strip() for s in sents if s.strip()]


# Embedding
def embed_sentences(sentences: list, embed_model, embed_batch_size: int) -> np.ndarray:
    """Return L2-normalised embeddings, shape (n, d)."""
    with torch.no_grad():
        embs = embed_model.encode(
            sentences,
            batch_size=embed_batch_size,
            normalize_embeddings=True,   # L2-normalise -> cosine = dot product
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    return embs


def cosine_similarity_matrix(emb_R: np.ndarray, emb_G: np.ndarray) -> np.ndarray:
    """emb_R (n,d), emb_G (m,d) already L2-normed -> returns (n,m)."""
    return emb_R @ emb_G.T


def adj_sim_matrix(sim: np.ndarray) -> np.ndarray:
    """ReLU: negative cosine -> 0."""
    return np.maximum(sim, 0.0)


# NLI
def nli_entailment_probs(
    premises: list,
    hypotheses: list,
    tokenizer,
    nli_model,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """P(entailment) for each (premise[i], hypothesis[i]) pair. Returns (N,)."""
    probs = []
    for i in range(0, len(premises), batch_size):
        batch_p = premises[i: i + batch_size]
        batch_h = hypotheses[i: i + batch_size]
        enc = tokenizer(
            batch_p, batch_h,
            padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = nli_model(**enc).logits          
        soft = F.softmax(logits.float(), dim=-1)
        probs.append(soft[:, ENTAILMENT_IDX].cpu().numpy())
    return np.concatenate(probs)


def compute_psym_matrix(
    R_sents: list,
    G_sents: list,
    tokenizer,
    nli_model,
    device: torch.device,
    nli_batch_size: int,
) -> np.ndarray:
    """
    Psym(r,g) = min(Pf(r->g), Pb(g->r))
    Returns (n, m) matrix.
    """
    n, m = len(R_sents), len(G_sents)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    # Flatten all n*m pairs
    prem_f = [R_sents[i] for i in range(n) for _ in range(m)]
    hyp_f  = [G_sents[k] for _ in range(n) for k in range(m)]

    Pf_flat = nli_entailment_probs(prem_f, hyp_f, tokenizer, nli_model, device, nli_batch_size)
    Pb_flat = nli_entailment_probs(hyp_f, prem_f, tokenizer, nli_model, device, nli_batch_size)

    Pf   = Pf_flat.reshape(n, m)
    Pb   = Pb_flat.reshape(n, m)
    Psym = np.minimum(Pf, Pb)
    return Psym.astype(np.float32)


# PropScore maths
def proposition_score_matrix(
    adj_sim: np.ndarray,
    psym:    np.ndarray,
    gamma:   float,
) -> np.ndarray:
    """
    PS(r,g) = [AdjSim * (1 + Psym) / 2] ^ (1 + gamma * (1 - Psym))
    """
    base     = adj_sim * (1.0 + psym) / 2.0
    exponent = 1.0 + gamma * (1.0 - psym)
    ps = np.where(base > 0, np.power(base, exponent), 0.0)
    return ps


def agg_ps(ps_row: np.ndarray, p: int) -> float:
    """
    AggPS(r_ij) = sum_{k=1}^{p}  t_k / log2(k + 1)
    Ranks are 1-based; log2(1+1)=1 at rank 1.
    """
    sorted_scores = np.sort(ps_row)[::-1]   # descending
    top_p = sorted_scores[:p]
    dcg = sum(float(score) / math.log2(k + 2)   # k 0-based -> log2(k+2)
              for k, score in enumerate(top_p))
    return dcg


# Per-item computation
def compute_propscore_for_item(
    item:            dict,
    embed_model,
    tokenizer,
    nli_model,
    device:          torch.device,
    embed_batch_size: int,
    nli_batch_size:  int,
) -> dict:
    ref_text = item.get("reference_answer", "")
    gen_text = item.get("generated_answer", "")

    R_sents = split_sentences(ref_text)
    G_sents = split_sentences(gen_text)

    n_i = len(R_sents)
    m_i = len(G_sents)
    p   = max(1, int(math.floor(math.sqrt(m_i))))

    # Degenerate case
    if n_i == 0 or m_i == 0:
        no_score  = [{f"gamma-{g}": 0.0} for g in GAMMAS]
        ref_props = {r: [{f"gamma-{g}": 0.0} for g in GAMMAS] for r in R_sents}
        return {
            "query_id":   item["query_id"],
            "question":   item["question"],
            "prop-score": no_score,
            "reference_answer_propositions": ref_props,
        }

    # Embeddings 
    emb_R = embed_sentences(R_sents, embed_model, embed_batch_size)   
    emb_G = embed_sentences(G_sents, embed_model, embed_batch_size)   

    sim_mat     = cosine_similarity_matrix(emb_R, emb_G)   
    adj_sim_mat = adj_sim_matrix(sim_mat)                  

    # NLI 
    psym_mat = compute_psym_matrix(
        R_sents, G_sents, tokenizer, nli_model, device, nli_batch_size
    )                                                        # (n, m)

    # Per-gamma computation 
    # agg_scores[j][g_idx] = AggPS(r_ij, gamma)
    agg_scores = [[0.0] * len(GAMMAS) for _ in range(n_i)]
    ps_avg     = [0.0] * len(GAMMAS)

    for g_idx, gamma in enumerate(GAMMAS):
        ps_mat   = proposition_score_matrix(adj_sim_mat, psym_mat, gamma)  # (n, m)
        agg_list = []
        for j in range(n_i):
            val = agg_ps(ps_mat[j], p)
            agg_scores[j][g_idx] = val
            agg_list.append(val)
        ps_avg[g_idx] = float(np.mean(agg_list))

    # Build output record
    prop_score_list = [{f"gamma-{g}": ps_avg[i]} for i, g in enumerate(GAMMAS)]

    ref_props_out = {}
    for j, r in enumerate(R_sents):
        ref_props_out[r] = [
            {f"gamma-{g}": agg_scores[j][i]}
            for i, g in enumerate(GAMMAS)
        ]

    return {
        "query_id":   item["query_id"],
        "question":   item["question"],
        "prop-score": prop_score_list,
        "reference_answer_propositions": ref_props_out,
    }


# Per-file processing
def process_file(
    input_path:       Path,
    output_path:      Path,
    embed_model,
    tokenizer,
    nli_model,
    device:           torch.device,
    embed_batch_size: int,
    nli_batch_size:   int,
):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for item in tqdm(data, desc=input_path.name, leave=False):
        out_item = compute_propscore_for_item(
            item,
            embed_model, tokenizer, nli_model, device,
            embed_batch_size, nli_batch_size,
        )
        results.append(out_item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(results)} items -> {output_path}")


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Sentence-level PropScore for all JSON files under input_dir."
    )
    parser.add_argument(
        "--input_dir", default="results_qwen",
        help="Root directory containing exp*/FILE.json files (default: results_qwen)",
    )
    parser.add_argument(
        "--output_dir", default="sentence_propscore_qwen",
        help="Root directory for output files (default: sentence_propscore_qwen)",
    )
    parser.add_argument(
        "--embed_batch_size", type=int, default=256,
        help="Batch size for bi-encoder embedding (default: 256)",
    )
    parser.add_argument(
        "--nli_batch_size", type=int, default=64,
        help="Batch size for NLI cross-encoder (default: 64)",
    )
    parser.add_argument(
        "--specific_file", default=None,
        help="Process only one relative path, e.g. exp5a/19_filler_7R_12S.json",
    )
    args = parser.parse_args()

   

    # Device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    # Load models
    print(f"\nLoading bi-encoder : {EMBED_MODEL}")
    embed_model =SentenceTransformer(EMBED_MODEL, device=str(device))
    embed_model.eval()

    print(f"Loading NLI model  : {NLI_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    nli_model = nli_model.to(device).eval()

    global ENTAILMENT_IDX
    ENTAILMENT_IDX = resolve_entailment_idx(nli_model)
    print(f"  Entailment label index : {ENTAILMENT_IDX}")
    print(f"  Label map              : {nli_model.config.id2label}\n")

    # Collect files
    input_root  = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if args.specific_file:
        all_files = [input_root / args.specific_file]
    else:
        all_files = sorted(input_root.rglob("*.json"))

    print(f"Found {len(all_files)} file(s) to process.\n")

    for inp in all_files:
        rel = inp.relative_to(input_root)
        out = output_root / rel
        if out.exists():
            print(f"  Skip (already exists): {out}")
            continue
        print(f"Processing: {inp}")
        process_file(
            inp, out,
            embed_model, tokenizer, nli_model, device,
            args.embed_batch_size, args.nli_batch_size,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()