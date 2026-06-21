import json
import os
import glob
import argparse
from vllm import LLM, SamplingParams


def run_inference(input_dir, output_dir, llm=None, sampling_params=None):
    """
    Runs vLLM inference on specified input and output directories.
    Processes each variation file individually and saves a matching output file.
    
    Args:
        input_dir      : Directory containing the prompt JSON files.
        output_dir     : Directory where the result JSON files will be saved.
        llm            : A pre-loaded vLLM LLM instance. If None, a new one is created.
        sampling_params: A pre-configured SamplingParams instance. If None, defaults are used.
    """
    
    # 1. Initialize model if not passed in
    if llm is None:
        print("No LLM instance provided. Loading Phi-3 into GPU memory...")
        llm = LLM(
            model="microsoft/Phi-3-mini-128k-instruct",
            max_model_len=32000, #Adjust this value based on your needs.
            trust_remote_code=True,
            gpu_memory_utilization=0.90 #Adjust this value based on your GPU's available memory. 0.90 means 90% of GPU memory will be used for the model.
        )

    if sampling_params is None:
        print("No SamplingParams provided. Using defaults...")
        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.83,
            max_tokens=512,
            stop=["\n---", "\nDocument:", "\nQuestion:", "---"]
        )

    
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

        # Extract prompts
        prompts = [entry['prompt'] for entry in dataset]
        print(f"  Generating answers for {len(prompts)} prompts...")

        # Run batched inference
        outputs = llm.generate(prompts, sampling_params)

        # Stitch generated answers back into the dataset
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            dataset[i]['generated_answer'] = generated_text

        # Preserve all metadata fields and rename answer -> reference_answer
        SKIP_KEYS = {"prompt", "answer", "answers", "question", "query_id", "generated_answer"}
        results = []
        for entry in dataset:
            # FIX: Use .get() to safely handle missing keys without throwing a KeyError
            record = {
                "query_id":         entry.get("query_id"),
                "question":         entry.get("question"),
            }
            
            # Add all experiment-specific metadata fields
            for k, v in entry.items():
                if k not in SKIP_KEYS:
                    record[k] = v
                    
            record["reference_answer"] = entry.get("answer") or entry.get("answers", "")
            record["generated_answer"] = entry.get("generated_answer", "")
            results.append(record)
            
        dataset = results
        
        # Save results
        with open(output_filepath, 'w') as f:
            json.dump(dataset, f, indent=4)

        print(f"  Saved → {output_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM inference with custom directories.")
    parser.add_argument("--input_dir",  type=str, required=True, help="Directory containing the prompt JSON files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the result JSON files will be saved.")
    args = parser.parse_args()

    run_inference(input_dir=args.input_dir, output_dir=args.output_dir)