import os
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model loading
model_id = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"  # Required for batch generation

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto"
)



def build_prompt(text):
    prompt = (
        f"Extract all the important, independent, atomic propositions (factual statements) "
        f"from the following text. List each proposition on a new line starting with a dash (-).\n\n"
        f"Text: {text}\n\nPropositions:\n"
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant that extracts atomic propositions from text. Return only the propositions."},
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def parse_propositions(response):
    propositions = [
        line.strip().lstrip('-').strip()
        for line in response.split('\n')
        if line.strip().startswith('-')
    ]
    if not propositions:
        propositions = [line.strip() for line in response.split('\n') if line.strip()]
    return propositions

def extract_propositions_batch(texts):
    """Run a batch of texts through the model in one GPU pass."""
    prompts = [build_prompt(t) for t in texts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the newly generated tokens (skip the prompt)
    input_len = inputs.input_ids.shape[1]
    responses = [
        tokenizer.decode(output[input_len:], skip_special_tokens=True)
        for output in outputs
    ]
    return [parse_propositions(r) for r in responses]


def process_directory(base_dir, output_dir, batch_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(base_dir):
        json_files = [f for f in files if f.endswith('.json')]
        if not json_files:
            continue

        rel_path = os.path.relpath(root, base_dir)
        out_sub_dir = os.path.join(output_dir, rel_path)
        os.makedirs(out_sub_dir, exist_ok=True)

        subdir_name = rel_path if rel_path != '.' else os.path.basename(base_dir)
        print(f"\n{'='*60}")
        print(f"Processing subdirectory: {subdir_name}")
        print(f"{'='*60}")

        for file in json_files:
            file_path = os.path.join(root, file)
            out_file_path = os.path.join(out_sub_dir, file)

            if os.path.exists(out_file_path):
                print(f"  [SKIP] {file} already processed.")
                continue

            print(f"  [FILE] {file}")
            with open(file_path, 'r') as f:
                data = json.load(f)

            items_meta = []
            ref_texts = []
            gen_texts = []
            for item in data:
                items_meta.append({
                    "query_id": item.get('query_id'),
                    "question": item.get('question'),
                })
                ref_texts.append(item.get('reference_answer', ''))
                gen_texts.append(item.get('generated_answer', ''))

            ref_props_all = []
            for i in range(0, len(ref_texts), batch_size):
                batch = ref_texts[i:i + batch_size]
                non_empty = [(j, t) for j, t in enumerate(batch) if t]
                props_batch = [''] * len(batch)
                if non_empty:
                    indices, texts = zip(*non_empty)
                    results = extract_propositions_batch(list(texts))
                    for idx, props in zip(indices, results):
                        props_batch[idx] = props
                ref_props_all.extend(props_batch)
                print(f"  [REF]  Batch {i // batch_size + 1}/{(len(ref_texts) + batch_size - 1) // batch_size} done")

            gen_props_all = []
            for i in range(0, len(gen_texts), batch_size):
                batch = gen_texts[i:i + batch_size]
                non_empty = [(j, t) for j, t in enumerate(batch) if t]
                props_batch = [''] * len(batch)
                if non_empty:
                    indices, texts = zip(*non_empty)
                    results = extract_propositions_batch(list(texts))
                    for idx, props in zip(indices, results):
                        props_batch[idx] = props
                gen_props_all.extend(props_batch)
                print(f"  [GEN]  Batch {i // batch_size + 1}/{(len(gen_texts) + batch_size - 1) // batch_size} done")

            results = [
                {
                    "query_id": meta["query_id"],
                    "question": meta["question"],
                    "reference_answer_propositions": ref_props if ref_props else [],
                    "generated_answer_propositions": gen_props if gen_props else [],
                }
                for meta, ref_props, gen_props in zip(items_meta, ref_props_all, gen_props_all)
            ]

            with open(out_file_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"  [DONE] {file} saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract atomic propositions from QA JSON files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input results directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output propositions directory")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing (default: 128)")
    args = parser.parse_args()


    process_directory(args.input_dir, args.output_dir, args.batch_size)
