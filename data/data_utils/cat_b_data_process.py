import json
import random

def inject_noise_documents(input_filepath, output_filepath, required_noise_count=30, relevant_fields=None):
    # Fallback if no fields are provided
    if relevant_fields is None:
        relevant_fields = [
            'gold', 
            'answers',
            'most_relevants',
            'medium_relevant',
            'relevants',
            'distractors'
            ]
        
    print(f"Loading {input_filepath}...")
    with open(input_filepath, 'r') as file:
        dataset = json.load(file)

    # 1. Build the Global Pool and Index Map
    global_pool = []
    entry_to_indices = {} 

    current_global_idx = 0
    for entry_idx, entry in enumerate(dataset):
        docs_in_entry = []
        
        # DYNAMIC SWEEP: Check the entry against every known field variation
        for field in relevant_fields:
            if field in entry:
                data = entry[field]
                if field == 'answers':
                    # Extract the string, and join them together with a space
                    flat_answers = [ans[0] for ans in data if len(ans) > 0]
                    text = " ".join(flat_answers)
                    docs_in_entry.append(text)
                # If it is an array of documents (like medium_relevant usually is)
                elif isinstance(data, list):
                    text=[d.get("text", "") for d in data ]
                    docs_in_entry.extend(text)
                # If it is a single document string
                elif isinstance(data, str) and data.strip():
                    docs_in_entry.append(data)
                    
        # Map indices to prevent cross-contamination for this specific entry
        entry_indices = set()
        for doc in docs_in_entry:
            global_pool.append(doc)
            entry_indices.add(current_global_idx)
            current_global_idx += 1
            
        entry_to_indices[entry_idx] = entry_indices
            
    print(f"Built global pool of {len(global_pool)} total documents.")

    # Create a master set of all available indices
    all_indices = set(range(len(global_pool)))

    # 2. Assign Noise to Each Entry using Fast Index Sampling
    for entry_idx, entry in enumerate(dataset):
        # Subtract the current entry's document indices from the master pool
        valid_idx_pool = list(all_indices - entry_to_indices[entry_idx])
        
        if len(valid_idx_pool) < required_noise_count:
            raise ValueError(f"Dataset too small! Need {required_noise_count}, found {len(valid_idx_pool)}.")
        
        # Fast integer sampling
        sampled_indices = random.sample(valid_idx_pool, required_noise_count)
        sampled_noise = [global_pool[i] for i in sampled_indices]
        
        # Final shuffle before injection
        random.shuffle(sampled_noise)
        entry['noise'] = sampled_noise

    # 3. Save the Updated Dataset
    print(f"Saving to {output_filepath}...")
    with open(output_filepath, 'w') as file:
        json.dump(dataset, file, indent=4)
    print("Done.\n")


if __name__ == "__main__":

    random.seed(42)
    fields_to_scrape = [ 
        'gold', 
        'answers',
        'most_relevants',
        'medium_relevant',
        'relevants',
        'distractors'
    ]

    # Process Category A (Document-Grounded)
    inject_noise_documents(
        input_filepath="data/raw/eli5_org.json", 
        output_filepath="data/processed/eli5_org_with_noise.json",
        relevant_fields=fields_to_scrape
    )