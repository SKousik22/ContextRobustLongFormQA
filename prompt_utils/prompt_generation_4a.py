import json
import os

# The exact instruction prompt for your LLM
INSTRUCTION = "You are a helpful assistant. Write comprehensive, self-contained and concise answers and explanations for the question. Base your explanation strictly on the provided context. Provide ONLY your final explanation. Do not generate any extra documents, follow-up questions, or separators. Do not hallucinate. Don't fail to answer."

def load_data(filepath):
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as file:
        return json.load(file)

def extract_answers(raw_answers):
    """
    Flatten all answers into a single merged string.
    Each element in raw_answers may itself be a list (e.g. [['ans1'], ['ans2'], ['ans3']]),
    so we flatten everything and join with a space.
    """
    sentences = []
    for item in raw_answers:
        if isinstance(item, list):
            sentences.extend(str(s).strip() for s in item)
        else:
            sentences.append(str(item).strip())
    return " ".join(s for s in sentences if s)

def generate_experiment_4a(dataset):
    """
    Experiment 4A: Impact of Gold Positioning (Using Random Noise)
    Shifts the Gold document across specific depth intervals within a context of random noise.
    """
    # The distinct configurations: Total Noise Docs -> Insertion Positions
    configurations = {
        9: [0, 4, 9],               # 0%, 50%, 100% depth
        19: [0, 4, 9, 14, 19],      # 0%, 25%, 50%, 75%, 100% depth
        29: [0, 3, 7, 10, 14, 18, 22, 25, 29] # ~12.5% intervals
    }
    
    # Create the output directory
    output_dir = "prompts/exp4a"
    os.makedirs(output_dir, exist_ok=True)

    # Outer loops: Total context size and the specific position index
    for total_S, positions in configurations.items():
        for pos in positions:
            case_prompts = []
            depth_pct = (pos / total_S) * 100 if total_S > 0 else 0
            
            for entry in dataset:
                query_id = entry.get('query_id', 'unknown')
                question = entry.get('question', '')

                # 1. Extract and merge answers
                raw_answers = entry.get('answers', [])
                answer = extract_answers(raw_answers)

                # 2. Safely Extract the Gold text
                gold_data = entry.get('gold', [])
                gold_text = " ".join([d.get("text", "") for d in gold_data if isinstance(d, dict) and "text" in d])
                
                # 2. Extract the Noise text 
                # ('noise' is already a list of strings thanks to your noise injection script)
                all_noise = entry.get('noise', [])
                
                # Slice the array to the required total context size
                S_docs = all_noise[:total_S]
                
                if len(S_docs) < total_S:
                    print(f"Warning: Entry {query_id} only has {len(S_docs)} noise docs, but {total_S} requested.")
                
                # 3. Inject Gold into the specific position
                # We use .copy() so we don't accidentally mutate the sliced array
                context_blocks = S_docs.copy()
                context_blocks.insert(pos, gold_text)
                
                # 4. Construct the layout
                context_str = "\n\n---\n\n".join(context_blocks)
                
                # Build the final prompt string
                prompt_text = f"{INSTRUCTION}\n\n{context_str}\n\nQuestion: {question}\nAnswer:"
                
                case_prompts.append({
                    "query_id": query_id,
                    "total_noise_docs": total_S,
                    "gold_position_index": pos,
                    "depth_percentage": round(depth_pct, 1),
                    "question": question,
                    "answer": answer,
                    "prompt": prompt_text
                })
                
            # Save this specific positioning case to its own isolated JSON file
            filename = f"{output_dir}/{total_S}_noise_docs_pos_{pos}.json"
            with open(filename, 'w') as file:
                json.dump(case_prompts, file, indent=4)
                
            print(f"Saved {len(case_prompts)} prompts to {filename}")

if __name__ == "__main__":
    # CRITICAL: Point this to the noise-injected dataset!
    input_file = 'data/processed/eli5_good_with_noise.json' 
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}. Please run the noise injection script first.")
    else:
        dataset = load_data(input_file)
        print("Generating Experiment 4A prompts (Positioning with Random Noise)...")
        generate_experiment_4a(dataset)
        print("\nAll 17 positioning cases generated successfully in the 'prompts/exp4a_noise/' directory.")