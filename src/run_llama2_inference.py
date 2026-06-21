import json
import os
import glob
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


model_id = "meta-llama/Llama-2-7b-chat-hf"


def run_inference(input_dir, output_dir, llm=None, sampling_params=None, tokenizer=None, max_tokens_filter=None):
    """
    Runs vLLM inference on specified input and output directories.
    Processes each variation file individually and saves a matching output file.

    Args:
        input_dir        : Directory containing the prompt JSON files.
        output_dir       : Directory where the result JSON files will be saved.
        llm              : A pre-loaded vLLM LLM instance. If None, a new one is created.
        sampling_params  : A pre-configured SamplingParams instance. If None, defaults are used.
        tokenizer        : A pre-loaded HuggingFace tokenizer for length filtering.
                           Required if max_tokens_filter is set.
        max_tokens_filter: If set, prompts exceeding this token count are skipped.
    """
    # 1. Initialize model if not passed in
    if llm is None:
        print("No LLM instance provided. Loading LLaMA-2 into GPU memory...")
        llm = LLM(
            model=model_id,
            max_model_len=4096,
            trust_remote_code=True,
            gpu_memory_utilization=0.9
        )

    if sampling_params is None:
        print("No SamplingParams provided. Using defaults...")
        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=512,
            stop=["\n---", "\nDocument:", "\nQuestion:", "---"]
        )

    if max_tokens_filter is not None and tokenizer is None:
        print("Warning: max_tokens_filter is set but no tokenizer was provided. Filter will be skipped.")

    
    # 2. Create output directory
    os.makedirs(output_dir, exist_ok=True)

   
    # 3. Fetch all JSON prompt files
    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

    if not json_files:
        print(f"Error: No JSON files found in {input_dir}.")
        return

    print(f"Found {len(json_files)} file(s) in {input_dir}.")


    # 4. Process each file one by one
    for filepath in json_files:
        filename = os.path.basename(filepath)
        output_filepath = os.path.join(output_dir, filename)

        # Crash recovery: skip if already processed
        if os.path.exists(output_filepath):
            print(f"  Skipping {filename} — already processed.")
            continue

        print(f"  Processing: {filename}...")

        with open(filepath, 'r') as f:
            dataset = json.load(f)

        
        # 5. Filter prompts by token length if required
        if max_tokens_filter is not None and tokenizer is not None:
            valid_entries = []
            skipped_count = 0

            for entry in dataset:
                token_length = len(tokenizer.encode(entry['prompt']))
                if token_length <= max_tokens_filter:
                    valid_entries.append(entry)
                else:
                    skipped_count += 1

            print(f"  Token filter ({max_tokens_filter} tokens): "
                  f"{len(valid_entries)} kept, {skipped_count} skipped.")
            dataset = valid_entries

        if not dataset:
            print(f"  No valid prompts remaining after filtering. Skipping {filename}.")
            continue

        # Extract prompts
        prompts = [entry['prompt'] for entry in dataset]
        print(f"  Generating answers for {len(prompts)} prompts...")

        # Run batched inference
        outputs = llm.generate(prompts, sampling_params)

        # Stitch generated answers back into the dataset
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            dataset[i]['generated_answer'] = generated_text

        # Select only the required fields before saving
        dataset = [
            {
                "experiment":       entry["experiment"],
                "query_id":         entry["query_id"],
                "prompt":           entry["prompt"],
                "generated_answer": entry["generated_answer"]
            }
            for entry in dataset
        ]

        # Save results — only the valid, filtered entries
        with open(output_filepath, 'w') as f:
            json.dump(dataset, f, indent=4)

        print(f"  Saved --> {output_filepath}")



# Standalone entry point (single experiment)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM inference with custom directories.")
    parser.add_argument("--input_dir",         type=str, required=True,
                        help="Directory containing the prompt JSON files.")
    parser.add_argument("--output_dir",        type=str, required=True,
                        help="Directory where the result JSON files will be saved.")
    parser.add_argument("--max_tokens_filter", type=int, default=4096,
                        help="Skip prompts exceeding this token count. Default: 4096.")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(model_id)

    run_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer=tok,
        max_tokens_filter=args.max_tokens_filter
    )