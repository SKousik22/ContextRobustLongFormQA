# ContextRobustnessLongFormQA

Evaluating long-form question answering under distractors, random noise, changing
context depth, and mixed-context conditions.

This project studies how irrelevant context affects long-form question answering on
ELI5. It builds controlled prompt suites containing relevant documents, distracting
documents, and random noise; runs several open language models with vLLM; and evaluates
their answers with lexical, semantic, and proposition-level metrics.

The project supports four generation models:

- Qwen2.5-7B-Instruct-AWQ
- Phi-3-mini-128k-instruct
- Falcon3-7B-Instruct-GPTQ-Int8
- Llama-2-7b-chat-hf

Evaluation includes BERTScore, Sentence-BERT cosine similarity, ROUGE-L, BLEU, PropScore/ Atomic PropScore, and SentenceScore/ Sentence PropScore. The ablation notebook also compares the metrics with
human annotations using Pearson correlation, Kendall tau, and bootstrap significance
tests.

## Repository layout

```text
power_of_noise_ELI5/
|-- ablation/
|   |-- ablation_compute_all_metrics_and_significance_test.ipynb
|   `-- merged_annotated_propositions.json
|-- data/
|   |-- data_utils/             # Query IDs, relevance grouping, and noise injection
|   |-- processed/              # Prompt-ready ELI5 datasets
|   `-- raw/                    # Source and intermediate ELI5 data
|-- eval_utils/
|   |-- compute_base_scores.py  # BERTScore, Sentence-BERT, and ROUGE-L
|   |-- compute_bleuscore.py    # Sentence-level BLEU
|   |-- compute_propscore.py    # Atomic proposition PropScore
|   `-- compute_sentencescore.py
|-- prompt_utils/               # Experiment 1a through 5b prompt generators
|-- shell/
|   `-- generate_prompts.sh
`-- src/
    |-- generate_propositions.py
    |-- *_wrapper.py            # Run all ten experiments with one model
    `-- run_*_inference.py      # Run a model on a selected prompt directory
```

Generated `prompts/`, `results/`, and metric directories are created at runtime.

## Experiments

The `a` experiments use `data/processed/eli5_good_with_noise.json`; the `b` experiments
use `data/processed/eli5_org_with_noise.json`.

| Experiment | Variable under study                                      | Configurations                                            |
| ---------- | --------------------------------------------------------- | --------------------------------------------------------- |
| 1a/1b      | Number of distractor documents before the gold document   | 0, 4, 9, 14, 19, 24, 29                                   |
| 2a/2b      | Gold-document depth among distractors                     | 9, 19, or 29 distractors at several insertion depths      |
| 3a/3b      | Number of random noise documents before the gold document | 0, 4, 9, 14, 19, 24, 29                                   |
| 4a/4b      | Gold-document depth among random noise                    | 9, 19, or 29 noise documents at several insertion depths  |
| 5a/5b      | Mixture of random noise (`R`) and distractors (`S`)   | Multiple `R/S` mixtures totaling 9, 19, or 29 documents |

Each prompt asks the model to answer from the supplied context only. Context blocks are
separated by `---`, and the relevant gold document is inserted according to the selected
experiment configuration.

## Requirements

Run the project from its repository root. Model inference is designed for Linux with an
NVIDIA GPU and a CUDA-compatible PyTorch installation. vLLM is not natively supported on
standard Windows installations; use Linux, WSL2, or a Linux compute server.

Recommended environment:

- Python 3.10 or 3.11
- A CUDA-capable GPU
- Sufficient GPU memory for 7B models and long contexts
- Git LFS or Hugging Face access where required

Create an environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the CUDA-specific PyTorch build recommended for your system before this command
when the default PyPI wheel does not match your CUDA environment.

Authenticate with Hugging Face before using gated models such as Llama 2:

```bash
huggingface-cli login
```

Model weights are downloaded when their scripts first run. Adjust batch sizes,
`max_model_len`, and `gpu_memory_utilization` in the scripts if GPU memory is limited.

## Data preparation

The repository already contains raw and processed JSON data. To rebuild the main
processing path, run:

```bash
python data/data_utils/add_qid.py
python data/data_utils/process_data.py
python data/data_utils/cat_a_data_process.py
python data/data_utils/cat_b_data_process.py
```

These scripts:

1. Add one-based `query_id` values to the calibrated ELI5 records.
2. group retrieved documents by relevance and select a gold document;
3. fill each record to 30 distractors using documents from other queries; and
4. inject 30 random noise documents with seed 42 for the category A and B datasets.

The prompt generators expect records containing fields such as:

```json
{
  "query_id": 1,
  "question": "...",
  "answers": [["..."]],
  "gold": [{"title": "...", "text": "..."}],
  "distractors": [{"title": "...", "text": "..."}],
  "noise": ["..."]
}
```

## Generate prompts

Generate all ten experiment suites:

```bash
bash shell/generate_prompts.sh
```

Alternatively, run a single experiment from the repository root:

```bash
python prompt_utils/prompt_generation_1a.py
```

Outputs are written to `prompts/exp1a/` through `prompts/exp5b/`. Each JSON record
contains the question, reference answer information, experiment metadata, and final
`prompt` string.

## Run inference

Choose one model wrapper to load the model once and process all available prompt suites:

```bash
python src/qwen25_wrapper.py
# or
python src/phi3_wrapper.py
python src/falcon3_wrapper.py
python src/llama2_wrapper.py
```

Default model settings are:

| Wrapper                | Hugging Face model                       | Maximum context |
| ---------------------- | ---------------------------------------- | --------------: |
| `qwen25_wrapper.py`  | `Qwen/Qwen2.5-7B-Instruct-AWQ`         |          28,000 |
| `phi3_wrapper.py`    | `microsoft/Phi-3-mini-128k-instruct`   |          32,000 |
| `falcon3_wrapper.py` | `tiiuae/Falcon3-7B-Instruct-GPTQ-Int8` |          28,000 |
| `llama2_wrapper.py`  | `meta-llama/Llama-2-7b-chat-hf`        |           4,096 |

All wrappers currently map `prompts/exp*/` to `results/exp*/`. Before running another
model, rename the previous output directory or edit the wrapper to use a model-specific
location such as `results_qwen/`, `results_phi3/`, `results_falcon3/`, or
`results_llama2/`. Otherwise, existing files may be skipped or mixed with another run.

To process only one prompt directory, use the corresponding inference script:

```bash
python src/run_qwen25_inference.py \
  --input_dir prompts/exp1a \
  --output_dir results_qwen/exp1a
```

Replace `qwen25` with `phi3`, `falcon3`, or `llama2` as needed. Llama 2 also accepts
`--max_tokens_filter` and defaults to 4096 tokens.

Inference outputs preserve experiment metadata and add:

```json
{
  "query_id": 1,
  "question": "...",
  "reference_answer": "...",
  "generated_answer": "..."
}
```

## Evaluation

The metric programs recursively process every JSON file below `--input_dir` and mirror
the relative directory structure under `--output_dir`. Existing output files are skipped.

### Baseline metrics

Compute BERTScore F1, Sentence-BERT cosine similarity, and ROUGE-L F1:

```bash
python eval_utils/compute_base_scores.py \
  --input_dir results_qwen \
  --output_dir base_scores_qwen \
  --device cuda \
  --batch_size 64
```

Defaults are `roberta-large` for BERTScore and `all-mpnet-base-v2` for Sentence-BERT.
The program also writes `empty_fields_log.json`.

### BLEU

```bash
python eval_utils/compute_bleuscore.py \
  --input_dir results_qwen \
  --output_dir bleu_scores_qwen
```

BLEU uses NLTK tokenization, sentence BLEU, and smoothing method 1.

### Atomic PropScore

First extract atomic propositions from the reference and generated answers:

```bash
python src/generate_propositions.py \
  --input_dir results_qwen \
  --output_dir propositions_qwen \
  --batch_size 32
```

This stage uses `Qwen/Qwen2.5-3B-Instruct`. Then compute Atomic PropScore:

```bash
python eval_utils/compute_propscore.py \
  --input_dir propositions_qwen \
  --output_dir propscore_qwen \
  --embed_batch_size 128 \
  --nli_batch_size 32
```

### Sentence PropScore

Sentence PropScore operates directly on complete reference and generated answers:

```bash
python eval_utils/compute_sentencescore.py \
  --input_dir results_qwen \
  --output_dir sentence_propscore_qwen \
  --embed_batch_size 256 \
  --nli_batch_size 64
```

Both PropScore variants use `BAAI/bge-large-en-v1.5` as the bi-encoder and
`cross-encoder/nli-deberta-v3-large` for entailment. Use `--specific_file` with either
script to process one relative result path while debugging or tuning memory usage:

```bash
python eval_utils/compute_propscore.py \
  --input_dir propositions_qwen \
  --output_dir propscore_qwen \
  --specific_file exp5a/19_filler_7R_12S.json
```

## Ablation and significance analysis

`ablation/merged_annotated_propositions.json` contains 200 annotated records with:

- `human_score`
- `reference_answer` and `generated_answer`
- `reference_answer_propositions` and `generated_answer_propositions`

Open `ablation/ablation_compute_all_metrics_and_significance_test.ipynb` in Jupyter or
Google Colab. The notebook computes all six metric families, correlations against the
human scores, and bootstrap comparisons between the strongest PropScore variants and
the baselines.

For Colab, upload the annotated JSON file to:

```text
/content/merged_annotated_propositions.json
```

Then run the notebook from top to bottom with a GPU runtime. Metric result notebooks,
JSON files, correlation tables, and significance-test tables are written under
`/content/`.

## Reproducibility notes

- Run commands from the repository root because paths are relative.
- Noise injection uses `random.seed(42)`. The distractor augmentation in
  `process_data.py` uses NumPy sampling without an explicit seed.
- GPU results and throughput may vary by CUDA, PyTorch, transformers, and vLLM version.
- Lower the embedding and NLI batch sizes if PropScore runs out of GPU memory.
- Llama 2 is gated on Hugging Face and requires accepting its license before download.
- No license file is currently included in this repository. Add one before distributing
  the code or data publicly.

## Acknowledgments

This project uses the ELI5 question-answering data and models hosted on Hugging Face.
Please follow the original dataset and model licenses and cite their associated work when
publishing results.
