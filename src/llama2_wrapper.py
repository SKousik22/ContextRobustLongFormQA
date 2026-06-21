import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from run_llama2_inference import run_inference


model_id= "meta-llama/Llama-2-7b-chat-hf"


# 1. Define all experiment input/output pairs
experiments = [
    # Category A
    ("prompts/exp1a", "results/exp1a"),
    ("prompts/exp2a", "results/exp2a"),
    ("prompts/exp3a",       "results/exp3a"),
    ("prompts/exp4a",   "results/exp4a"),
    ("prompts/exp5a",    "results/exp5a"),
    # Category B
    ("prompts/exp1b", "results/exp1b"),
    ("prompts/exp2b", "results/exp2b"),
    ("prompts/exp3b",       "results/exp3b"),
    ("prompts/exp4b",   "results/exp4b"),
    ("prompts/exp5b",    "results/exp5b"),
]


# 2. Load model, tokenizer, and sampling params ONCE
print("=" * 60)
print("Loading LLaMA-2 and tokenizer into GPU memory (one-time load)...")
print("=" * 60)

llm = LLM(
    model=model_id,
    max_model_len=4096,
    trust_remote_code=True,
    gpu_memory_utilization=0.9
)

sampling_params = SamplingParams(
    temperature=0.3,
    top_p=0.9,
    max_tokens=512,
    stop=["\n---", "\nDocument:", "\nQuestion:", "---"]
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Model and tokenizer loaded successfully. Starting experiments...\n")



# 3. Run all experiments sequentially
total= len(experiments)
completed = 0
skipped = 0
failed= 0

for i, (input_dir, output_dir) in enumerate(experiments, start=1):
    exp_name = input_dir.split("/")[-1]

    print(f"[{i}/{total}] Starting: {exp_name}")
    print(f"            Input  --> {input_dir}")
    print(f"            Output --> {output_dir}")

    if not os.path.exists(input_dir):
        print(f"            SKIPPED — input directory not found.\n")
        skipped += 1
        continue

    try:
        run_inference(
            input_dir=input_dir,
            output_dir=output_dir,
            llm=llm,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            max_tokens_filter=4096
        )
        completed += 1
        print(f"            DONE — results saved to {output_dir}")
    except Exception as e:
        failed += 1
        print(f"            FAILED — {e}")

    print(f"            Progress: {completed} completed, {skipped} skipped, "
          f"{failed} failed out of {total} total.\n")


# 4. Final summary
print("=" * 60)
print("ALL EXPERIMENTS FINISHED")
print(f"   Completed : {completed}")
print(f"   Failed    : {failed}")
print(f"   Skipped   : {skipped}")
print("=" * 60)