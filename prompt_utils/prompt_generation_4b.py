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

def generate_experiment_4b(dataset):
    configurations = {
        9: [0, 4, 9],               
        19: [0, 4, 9, 14, 19],      
        29: [0, 3, 7, 10, 14, 18, 22, 25, 29] 
    }
    
    output_dir = "prompts/exp4b"
    os.makedirs(output_dir, exist_ok=True)

    for total_S, positions in configurations.items():
        for pos in positions:
            case_prompts = []
            depth_pct = (pos / total_S) * 100 if total_S > 0 else 0
            
            for entry in dataset:
                query_id = entry.get('query_id', entry.get('id', 'unknown'))
                question = entry.get('question', entry.get('query', ''))
                
                raw_answers = entry.get('answers') or entry.get('answer') or []
                answer = extract_answers(raw_answers)
                
                all_noise = entry.get('noise', [])
                
                S_docs = all_noise[:total_S]
                
                if len(S_docs) < total_S:
                    print(f"Warning: Entry {query_id} only has {len(S_docs)} noise docs, but {total_S} requested.")
                
                context_blocks = S_docs.copy()
                context_blocks.insert(pos, answer)
                
                context_str = "\n\n---\n\n".join(context_blocks)
                prompt_text = f"{INSTRUCTION}\n\n{context_str}\n\nQuestion: {question}\nAnswer:"
                
                case_prompts.append({
                    "query_id": query_id,
                    "total_noise_docs": total_S,
                    "gold_position_index": pos, # FIXED: Key matched to 4a
                    "depth_percentage": round(depth_pct, 1),
                    "question": question,       # FIXED: Capitalization matched to 4a
                    "answer": answer,
                    "prompt": prompt_text
                })
                
            filename = f"{output_dir}/{total_S}_noise_docs_pos_{pos}.json"
            with open(filename, 'w') as file:
                json.dump(case_prompts, file, indent=4)
                
            print(f"Saved {len(case_prompts)} prompts to {filename}")

if __name__ == "__main__":
    input_file = 'data/processed/eli5_org_with_noise.json' 
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}. Please run the noise injection script first.")
    else:
        dataset = load_data(input_file)
        print("Generating Experiment 4B prompts (Positioning with Random Noise)...")
        generate_experiment_4b(dataset)
        print("\nAll 17 positioning cases generated successfully in the 'prompts/exp4b_noise/' directory.")