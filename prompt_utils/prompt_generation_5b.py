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

def generate_experiment_5b(dataset):
    configurations = {
        9: [
            (0, 9), (2, 7), (5, 4), (7, 2), (9, 0)
        ],
        19: [
            (0, 19), (2, 17), (4, 15), (7, 12), (9, 10), 
            (11, 8), (14, 5), (17, 2), (19, 0)
        ],
        29: [
            (0, 29), (2, 27), (4, 25), (7, 22), (9, 20), 
            (12, 17), (14, 15), (17, 12), (19, 10), (22, 7), 
            (24, 5), (27, 2), (29, 0)
        ]
    }
    
    output_dir = "prompts/exp5b"
    os.makedirs(output_dir, exist_ok=True)

    for total_filler, ratios in configurations.items():
        for r_count, s_count in ratios:
            case_prompts = []
            
            for entry in dataset:
                query_id = entry.get('query_id', entry.get('id', 'unknown'))
                question = entry.get('question', entry.get('query', ''))
                
                raw_answers = entry.get('answers') or entry.get('answer') or []
                answer = extract_answers(raw_answers)
                
                distractor_data = entry.get('distractors') or entry.get('distractor') or []
                all_distractors = [d.get("text", "") for d in distractor_data if isinstance(d, dict) and "text" in d]
                
                all_noise = entry.get('noise', [])
                
                if len(all_noise) < r_count:
                    print(f"Warning: Entry {query_id} only has {len(all_noise)} noise docs, but {r_count} requested.")
                if len(all_distractors) < s_count:
                    print(f"Warning: Entry {query_id} only has {len(all_distractors)} distractors, but {s_count} requested.")
                
                R_docs = all_noise[:r_count]
                S_docs = all_distractors[:s_count]
                
                context_blocks = R_docs + S_docs + [answer]
                context_str = "\n\n---\n\n".join(context_blocks)
                
                prompt_text = f"{INSTRUCTION}\n\n{context_str}\n\nQuestion: {question}\nAnswer:"
                
                case_prompts.append({
                    "query_id": query_id,
                    "total_filler": total_filler,
                    "noise_R_count": r_count,
                    "distractor_S_count": s_count,
                    "question": question, # FIXED: Capitalization matched to 5a
                    "answer": answer,
                    "prompt": prompt_text
                })
                
            filename = f"{output_dir}/{total_filler}_filler_{r_count}R_{s_count}S.json"
            with open(filename, 'w') as file:
                json.dump(case_prompts, file, indent=4)
                
            print(f"Saved {len(case_prompts)} prompts to {filename}")

if __name__ == "__main__":
    input_file = 'data/processed/eli5_org_with_noise.json' 
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}. Please run the noise injection script first.")
    else:
        dataset = load_data(input_file)
        print("Generating Experiment 5B prompts (Composition Tradeoff - Answer Grounded)...")
        generate_experiment_5b(dataset)
        print("\nAll composition cases generated successfully in the 'prompts/exp5b/' directory.")