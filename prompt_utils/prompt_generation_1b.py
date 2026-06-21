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

def generate_experiment_1b_distractors(dataset):
    """
    Experiment 1B: Impact of Distracting Documents
    Layout: [ Instruction, N x R (Distractors), G (Gold), Question ]
    """
    # The different context lengths (number of distractor documents)
    distractor_counts = [0, 4, 9, 14, 19, 24, 29]
    
    # Create the output directory
    output_dir = "prompts/exp1b"
    os.makedirs(output_dir, exist_ok=True)

    for count in distractor_counts:
        case_prompts = []
        
        for entry in dataset:
            query_id = entry.get('query_id', 'unknown')
            question = entry.get('question', '')
            
            # 1. Extract and merge answers
            raw_answers = entry.get('answers', [])
            answer = extract_answers(raw_answers)
            
            # 2. Extract the Distractors text (up to the current count)
            distractor_data = entry.get('distractors', [])
            R_docs = [d.get("text", "") for d in distractor_data if isinstance(d, dict) and "text" in d]
            
            # Slice the array to the required number of distractors
            R_docs = R_docs[:count]
            
            if len(R_docs) < count:
                print(f"Warning: Entry {query_id} only has {len(R_docs)} distractors, but {count} requested.")
            
            # 3. Construct layout: R distractor documents followed by the Gold document
            context_blocks = R_docs + [answer]
            context_str = "\n\n---\n\n".join(context_blocks)
            
            # 4. Build the final prompt string
            prompt_text = f"{INSTRUCTION}\n\n{context_str}\n\nQuestion: {question}\nAnswer:"
            
            case_prompts.append({
                "query_id": query_id,
                "distractor_count": count,
                "question": question,
                "answer": answer,
                "prompt": prompt_text
            })
            
        # Save this specific case to its own isolated JSON file
        filename = f"{output_dir}/{count}_distractors.json"
        with open(filename, 'w') as file:
            json.dump(case_prompts, file, indent=4)
            
        print(f"Saved {len(case_prompts)} prompts to {filename}")

if __name__ == "__main__":
    # You can point this directly at your original eli5_good.json 
    # since it already contains the distractors!
    input_file = 'data/processed/eli5_org_with_noise.json' 
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}.")
    else:
        dataset = load_data(input_file)
        print("Generating Experiment 1B prompts (using Distractors)...")
        generate_experiment_1b_distractors(dataset)
        print("\nAll cases generated successfully in the 'exp_1b/prompts' directory.")