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
    # 1. Initialize model if not passed in
    if llm is None:
        print("No LLM instance provided. Loading full, unquantized Qwen2.5 into GPU memory...")
        
        model_id="Qwen/Qwen2.5-7B-Instruct" 
        
        llm =LLM(
            model=model_id,
            dtype="auto",           #Change this to "awq", or "gptq" if you want to use a quantized model
            max_model_len=28000,    #Adjust this value based on your needs.
            trust_remote_code=True,
            gpu_memory_utilization=0.90 # Adjust this value based on your GPU's available memory. 0.90 means 90% of GPU memory will be used for the model.
        )

    if sampling_params is None:
        print("No SamplingParams provided. Using defaults...")
        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=512,
        )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all JSON files in the input directory
    input_files = glob.glob(os.path.join(input_dir, "*.json"))
    if not input_files:
        print(f"  No JSON files found in {input_dir}")
        return

    for input_filepath in input_files:
        filename = os.path.basename(input_filepath)
        output_filepath = os.path.join(output_dir, filename)


        #Crash recovery: skip if output file already exists
        if os.path.exists(output_filepath):
            print(f"  Skipping {filename} — already processed.")
            continue

        print(f"  Processing {filename}...")

        # Load input prompts
        with open(input_filepath, 'r') as f:
            dataset = json.load(f)

        # Extract prompts
        prompts = [entry['prompt'] for entry in dataset]
        print(f"  Generating answers for {len(prompts)} prompts...")

        # Run batched inference
        outputs = llm.generate(prompts, sampling_params)

        # Stitch generated answers back into the dataset
        for i, output in enumerate(outputs):
            dataset[i]['generated_answer'] = output.outputs[0].text.strip()

        # Preserve all metadata fields and rename answer -> reference_answer
        SKIP_KEYS = {"prompt", "answer", "answers", "question", "query_id", "generated_answer"}
        results = []
        for entry in dataset:
            record = {
                "query_id":         entry["query_id"],
                "question":         entry["question"],
            }
            # Add all experiment-specific metadata fields (e.g. distractor_count, depth_percentage, etc.)
            for k, v in entry.items():
                if k not in SKIP_KEYS:
                    record[k] = v
            record["reference_answer"] = entry.get("answer") or entry.get("answers", "")
            record["generated_answer"] = entry["generated_answer"]
            results.append(record)
        dataset = results
        
        # Save results
        with open(output_filepath, 'w') as f:
            json.dump(dataset, f, indent=4)

        print(f"  Saved --> {output_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM inference with custom directories.")
    parser.add_argument("--input_dir",  type=str, required=True, help="Directory containing the prompt JSON files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the result JSON files will be saved.")
    args = parser.parse_args()

    run_inference(input_dir=args.input_dir, output_dir=args.output_dir)