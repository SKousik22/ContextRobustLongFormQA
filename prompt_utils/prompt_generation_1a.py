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

def generate_experiment_1a_distractors(dataset):
    """
    Experiment 1A: Impact of Distracting Documents
    Layout: [ Instruction, N x S (Distractors), G (Gold), Question ]

    Each output record contains:
        - query_id
        - distractor_count
        - question
        - answers        (list of up to 3 answer strings)
        - prompt
    """
    distractor_counts = [0, 4, 9, 14, 19, 24, 29]

    output_dir = "prompts/exp1a"
    os.makedirs(output_dir, exist_ok=True)

    for count in distractor_counts:
        case_prompts = []

        for entry in dataset:
            query_id  = entry.get('query_id', 'unknown')
            question  = entry.get('question', '')

            # --- Answers ---
            raw_answers = entry.get('answers', [])
            answers = extract_answers(raw_answers)

            # --- Gold document ---
            gold_data = entry.get('gold', [])
            gold_text = " ".join(
                d.get("text", "") for d in gold_data
                if isinstance(d, dict) and "text" in d
            )

            # --- Distractor documents (sliced to current count) ---
            distractor_data = entry.get('distractors', [])
            S_docs = [
                d.get("text", "") for d in distractor_data
                if isinstance(d, dict) and "text" in d
            ]
            S_docs = S_docs[:count]

            if len(S_docs) < count:
                print(
                    f"Warning: Entry {query_id} only has {len(S_docs)} distractors, "
                    f"but {count} requested."
                )

            # --- Build prompt ---
            context_blocks = S_docs + [gold_text]
            context_str    = "\n\n---\n\n".join(context_blocks)
            prompt_text    = f"{INSTRUCTION}\n\n{context_str}\n\nQuestion: {question}\nAnswer:"

            case_prompts.append({
                "query_id":        query_id,
                "distractor_count": count,
                "question":        question,
                "answers":         answers,
                "prompt":          prompt_text,
            })

        filename = f"{output_dir}/{count}_distractors.json"
        with open(filename, 'w') as file:
            json.dump(case_prompts, file, indent=4)

        print(f"Saved {len(case_prompts)} prompts to {filename}")

if __name__ == "__main__":
    input_file = 'data/processed/eli5_good_with_noise.json'

    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}.")
    else:
        dataset = load_data(input_file)
        print("Generating Experiment 1A prompts (using Distractors)...")
        generate_experiment_1a_distractors(dataset)
        print("\nAll cases generated successfully in the 'prompts/exp1a/' directory.")