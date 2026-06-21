import os
from vllm import LLM, SamplingParams
from run_qwen25_inference import run_inference


# 1. Define all experiment input/output pairs
experiments = [
    # Category A
    ("prompts/exp1a",       "results/exp1a"),
    ("prompts/exp2a",       "results/exp2a"),
    ("prompts/exp3a",       "results/exp3a"),
    ("prompts/exp4a",       "results/exp4a"),
    ("prompts/exp5a",       "results/exp5a"),

    # Category B
    ("prompts/exp1b",       "results/exp1b"),
    ("prompts/exp2b",       "results/exp2b"),
    ("prompts/exp3b",       "results/exp3b"),
    ("prompts/exp4b",       "results/exp4b"),
    ("prompts/exp5b",       "results/exp5b"),
]


# 2. Load model and sampling params

print("=" * 60)
print("Loading local Qwen2.5-7B-Instruct into GPU memory ...")
print("=" * 60)

model_id= "Qwen/Qwen2.5-7B-Instruct-AWQ"

llm = LLM(
    model=model_id,
    dtype="auto",           #Change this to "awq", or "gptq" if you want to use a quantized model
    max_model_len=28000,    #Adjust this value based on your needs.
    trust_remote_code=True,
    gpu_memory_utilization=0.90 # Adjust this value based on your GPU's available memory. 0.90 means 90% of GPU memory will be used for the model.
    
)

sampling_params = SamplingParams(
    temperature=0.3,
    top_p=0.9,
    max_tokens=512,
    stop=["\n---", "\nDocument:", "\nQuestion:", "---"]
)

print("Model loaded successfully. Starting experiments...\n")


# 3. Run all experiments sequentially

total = len(experiments)
completed = 0
skipped = 0
failed = 0

for i, (input_dir, output_dir) in enumerate(experiments, start=1):
    exp_name = input_dir.split("/")[-1]  
    
    print(f"[{i}/{total}] Starting: {exp_name}")
    print(f"            Input  --> {input_dir}")
    print(f"            Output --> {output_dir}")

    if not os.path.exists(input_dir):
        print(f"        SKIPPED — input directory not found.\n")
        skipped+=1
        continue

    try:
        run_inference(
            input_dir=input_dir,
            output_dir=output_dir,
            llm=llm,
            sampling_params=sampling_params
        )
        completed+=1
        print(f"        DONE — results saved to {output_dir}\n")
    except Exception as e:
        failed+=1
        print(f"        FAILED — Error processing experiment {exp_name}: {e}\n")

print("=" * 60)
print("EXPERIMENT RUN COMPLETE SUMMARY:")
print(f"  Total Configurations: {total}")
print(f"  Successfully Done : {completed}")
print(f"  Skipped (No Input) : {skipped}")
print(f"  Failed Run Errors : {failed}")
print("=" * 60)