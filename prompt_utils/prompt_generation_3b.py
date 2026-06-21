import json
import os

INSTRUCTION = "You are a helpful assistant. Write comprehensive, self-contained and concise answers and explanations for the question. Base your explanation strictly on the provided context. Provide ONLY your final explanation. Do not generate any extra documents, follow-up questions, or separators. Do not hallucinate. Don't fail to answer."

def load_data(filepath):
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as file:
        return json.load(file)

def extract_answers(raw_answers):
    sentences = []
    for item in raw_answers:
        if isinstance(item, list):
            sentences.extend(str(s).strip() for s in item)
        else:
            sentences.append(str(item).strip())
    return " ".join(s for s in sentences if s)

def generate_experiment_3b(dataset):
    noise_counts = [0, 4, 9, 14, 19, 24, 29]
    
    output_dir = "prompts/exp3b"
    os.makedirs(output_dir, exist_ok=True)

    for count in noise_counts:
        case_prompts = []
        
        for entry in dataset:
            query_id = entry.get('query_id', entry.get('id', 'unknown'))
            question = entry.get('question', entry.get('query', ''))
            
            raw_answers = entry.get('answers', entry.get('answer', []))
            answer = extract_answers(raw_answers)
            
            noise_docs = entry.get('noise', [])
            
            if len(noise_docs) < count:
                print(f"Warning: Entry {query_id} only has {len(noise_docs)} noise docs, but {count} requested.")
                
            S_docs = noise_docs[:count]
            
            context_blocks = S_docs + [answer]
            context_str = "\n\n---\n\n".join(context_blocks)
            prompt_text = f"{INSTRUCTION}\n\n{context_str}\n\nQuestion: {question}\nAnswer:"
            
            case_prompts.append({
                "query_id": query_id,
                "noise_count": count,
                "question": question, # FIXED: Capitalization matched to 3a
                "answer": answer,
                "prompt": prompt_text
            })
            
        filename = f"{output_dir}/{count}_noise_docs.json"
        with open(filename, 'w') as file:
            json.dump(case_prompts, file, indent=4)
            
        print(f"Saved {len(case_prompts)} prompts to {filename}")

if __name__ == "__main__":
    input_file = 'data/processed/eli5_org_with_noise.json' 
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}. Make sure you ran the noise injection script first.")
    else:
        dataset = load_data(input_file)
        print("Generating Experiment 3B prompts (using Random Noise)...")
        generate_experiment_3b(dataset)
        print("\nAll cases generated successfully in the 'prompts/exp3b_noise/' directory.")