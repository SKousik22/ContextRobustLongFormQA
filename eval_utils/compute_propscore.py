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
      --output_dir propscore_qwen \
      [--batch_size 64] [--nli_batch_size 32] [--workers 4]

"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Constants
GAMMAS = list(range(1, 16))         
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
NLI_MODEL   = "cross-encoder/nli-deberta-v3-large"


ENTAILMENT_IDX = None   # resolved at runtime, if you need fixed index, set it here (e.g. 0 for MNLI-based models)


# Helpers
def resolve_entailment_idx(model):
    """Return the logit column that corresponds to 'entailment'."""
    id2label = model.config.id2label
    for idx, label in id2label.items():
        if label.lower() == "entailment":
            return int(idx)
    raise ValueError(f"'entailment' not found in id2label: {id2label}")


def cosine_sim_matrix(emb_r: np.ndarray, emb_g: np.ndarray) -> np.ndarray:
    """
    emb_r : (n, d)   reference embeddings
    emb_g : (m, d)   generated embeddings
    Returns (n, m) cosine similarity matrix.
    """
    # Normalise
    r_norm = emb_r / (np.linalg.norm(emb_r, axis=1, keepdims=True) + 1e-12)
    g_norm = emb_g / (np.linalg.norm(emb_g, axis=1, keepdims=True) + 1e-12)
    return r_norm @ g_norm.T          


def adj_sim_matrix(sim: np.ndarray) -> np.ndarray:
    """ReLU-adjusted similarity: max(sim, 0.0)"""
    return np.maximum(sim, 0.0)


def nli_entailment_probs(
    premises: list[str],
    hypotheses: list[str],
    tokenizer,
    nli_model,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """
    For each (premise[i], hypothesis[i]) pair return the softmax
    probability of the 'entailment' class.
    """
    global ENTAILMENT_IDX
    probs = []
    for start in range(0, len(premises), batch_size):
        batch_p=premises[start: start + batch_size]
        batch_h=hypotheses[start: start + batch_size]
        enc = tokenizer(
            batch_p, batch_h,
            padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = nli_model(**enc).logits          
        soft = F.softmax(logits.float(), dim=-1)
        probs.append(soft[:, ENTAILMENT_IDX].cpu().numpy())
    return np.concatenate(probs)


def compute_psym_matrix(
    refs: list[str],
    gens: list[str],
    tokenizer,
    nli_model,
    device: torch.device,
    nli_batch_size: int,
) -> np.ndarray:
    """
    Returns Psym matrix of shape (n, m).
    Psym(r,g) = min(Pf(r->g), Pb(g->r))
    """
    n, m = len(refs), len(gens)
    if n == 0 or m == 0:
        return np.zeros((n, m))

    # Flatten all pairs
    all_premises_f   = [refs[i] for i in range(n) for _ in range(m)]
    all_hypotheses_f = [gens[k] for _ in range(n) for k in range(m)]

    pf_flat = nli_entailment_probs(
        all_premises_f, all_hypotheses_f,
        tokenizer, nli_model, device, nli_batch_size
    )

    # Reverse direction
    pb_flat = nli_entailment_probs(
        all_hypotheses_f, all_premises_f,
        tokenizer, nli_model, device, nli_batch_size
    )

    pf = pf_flat.reshape(n, m)
    pb = pb_flat.reshape(n, m)
    return np.minimum(pf, pb)          


def proposition_score_matrix(
    adj_sim: np.ndarray,    
    psym:    np.ndarray,    
    gamma:   float,
) -> np.ndarray:
    """
    PS(r,g) = [ AdjSim * (1 + Psym) / 2 ] ^ (1 + gamma * (1 - Psym))
    """
    base     = adj_sim * (1.0 + psym) / 2.0
    exponent = 1.0 + gamma * (1.0 - psym)
    ps = np.where(base > 0, np.power(base, exponent), 0.0)
    return ps


def agg_ps(ps_row: np.ndarray, p: int) -> float:
    """
    AggPS(r_ij) = sum_{k=1}^{p}  t_k / log2(k+1)
    where t_1 >= t_2 >= ... are the sorted PS scores.
    """
    sorted_scores = np.sort(ps_row)[::-1]   # descending
    top_p = sorted_scores[:p]
    dcg = sum(score / math.log2(k + 2)      
              for k, score in enumerate(top_p))
    return float(dcg)


def prop_score(agg_list: list[float]) -> float:
    if not agg_list:
        return 0.0
    return float(np.mean(agg_list))


# ─────────────────────────────────────────────────────────────
# Per-file processing
# ─────────────────────────────────────────────────────────────

def process_file(
    input_path: Path,
    output_path: Path,
    embed_model,
    tokenizer,
    nli_model,
    device: torch.device,
    embed_batch_size: int,
    nli_batch_size: int,
):
    with open(input_path) as f:
        items = json.load(f)

    output_items = []

    for item in tqdm(items, desc=str(input_path.name), leave=False):
        qid = item["query_id"]
        question = item["question"]
        refs = item["reference_answer_propositions"]    # R_i
        gens= item["generated_answer_propositions"]    # G_i

        n=len(refs)
        m =len(gens)
        p= max(1, math.ceil(math.sqrt(m)))  #Adjust this if you want a different top-p strategy

        #Embeddings
        all_texts =refs + gens
        embeddings= embed_model.encode(
            all_texts,
            batch_size=embed_batch_size,
            normalize_embeddings=False,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        emb_r=embeddings[:n]
        emb_g =embeddings[n:]

        sim= cosine_sim_matrix(emb_r, emb_g)
        adj_sim =adj_sim_matrix(sim)   

        #NLI
        psym = compute_psym_matrix(
            refs, gens, tokenizer, nli_model, device, nli_batch_size
        )                                            

        # Per-gamma scores 
        # ref_agg[j][gamma_idx] = AggPS(r_ij, gamma)
        ref_agg  = [[0.0] * len(GAMMAS) for _ in range(n)]
        ps_avg   = [0.0] * len(GAMMAS)

        for g_idx, gamma in enumerate(GAMMAS):
            ps_mat = proposition_score_matrix(adj_sim, psym, gamma)  
            agg_list = []
            for j in range(n):
                val = agg_ps(ps_mat[j], p)
                ref_agg[j][g_idx] = val
                agg_list.append(val)
            ps_avg[g_idx] = prop_score(agg_list)

        # Build output record 
        prop_score_list = [{f"gamma-{g}": ps_avg[i]} for i, g in enumerate(GAMMAS)]

        ref_agg_out = {}
        for j, r_text in enumerate(refs):
            ref_agg_out[r_text] = [
                {f"gamma-{g}": ref_agg[j][i]}
                for i, g in enumerate(GAMMAS)
            ]

        output_items.append({
            "query_id":   qid,
            "question":   question,
            "prop-score": prop_score_list,
            "reference_answer_propositions": ref_agg_out,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_items, f, indent=2, ensure_ascii=False)

    print(f"   Saved {len(output_items)} items --> {output_path}")



# Main
def main():
    parser = argparse.ArgumentParser(description="Compute PropScore for all JSON files.")
    parser.add_argument("--input_dir",       default="results_qwen",   help="Root input directory")
    parser.add_argument("--output_dir",      default="propscore_qwen", help="Root output directory")
    parser.add_argument("--embed_batch_size", type=int, default=128,   help="Embedding batch size")
    parser.add_argument("--nli_batch_size",   type=int, default=32,    help="NLI inference batch size")
    parser.add_argument("--specific_file",   default=None,
                        help="Optional: process only this relative path, e.g. exp5a/19_filler_7R_12S.json")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        # Enable TF32 for Ampere+ / Ada / Blackwell (RTX 50xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load models
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    embed_model = SentenceTransformer(EMBED_MODEL, device=str(device))

    print(f"Loading NLI model: {NLI_MODEL}")
    tokenizer  = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli_model  = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    nli_model  = nli_model.to(device).eval()

    global ENTAILMENT_IDX
    ENTAILMENT_IDX = resolve_entailment_idx(nli_model)
    print(f"  Entailment label index: {ENTAILMENT_IDX}")
    print(f"  Label map: {nli_model.config.id2label}")

    # Collect files
    input_root  = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if args.specific_file:
        files = [input_root / args.specific_file]
    else:
        files = sorted(input_root.rglob("*.json"))

    print(f"\nFound {len(files)} file(s) to process.\n")

    for inp in files:
        rel        = inp.relative_to(input_root)
        out        = output_root / rel
        if out.exists():
            print(f"  >>  Skip (exists): {out}")
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
