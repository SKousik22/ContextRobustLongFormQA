import json
import numpy as np
from typing import List, Dict, Any

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
            if isinstance(data, dict):
                return [data]
            return data
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]
            return data

def segregate(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segregated_data = []
    for item in data:
        question = item.get("question", "")
        # Preserve other keys if needed, e.g., query_id
        query_id = item.get("query_id") 
        docs = item.get("docs", [])
        answers = item.get('answers', [])

        gold = []
        most_relevants = []
        medium_relevants = []
        relevants = []
        distractors = []

        for doc in docs:
            title = doc.get("title", "")
            text = doc.get("text", "")
            answers_found = doc.get("answers_found", [])
            rec_score = doc.get("rec_score", 0)

            doc_obj = {
                "title": title,
                "text": text,
            }

            if rec_score >= 90 or sum(answers_found) == 3:
                most_relevants.append(doc_obj)
            elif rec_score >= 50 or sum(answers_found) >= 2:
                medium_relevants.append(doc_obj)
            elif rec_score >= 20 or sum(answers_found) >= 1:
                relevants.append(doc_obj)
            else:
                distractors.append(doc_obj)
            

        
        #Fix gold document
        if most_relevants:
            gold.append(most_relevants[0])
        elif medium_relevants:
            gold.append(medium_relevants[0])
        elif relevants:
            gold.append(relevants[0])
            

        
        segregated_item = {
            "query_id": query_id,
            "question": question,
            "answers" : answers,
            "gold": gold,
            "most_relevants": most_relevants,
            "medium_relevant" : medium_relevants,
            "relevants": relevants,
            "distractors": distractors
        }
        segregated_data.append(segregated_item)
    
    return segregated_data

def add_distractors(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Note: 'data' here must be the output of segregate()
    final_data = []
    
    # Valid indices are 0 to len(data)-1
    low = 0
    high = len(data)
    
    for i, ex in enumerate(data):
        # Create a deep-ish copy to avoid modifying original list in place
        item = ex.copy()
        
        # Ensure the list exists so we can check length and append
        current_distractors = item.setdefault('distractors', [])
        num_docs = len(current_distractors)
        
        # Target is to fill up to 30 distractors
        target_count = 30 
        
        if num_docs < target_count:
            needed = target_count - num_docs
            arr = []
            
            # Simple rejection sampling
            while len(arr) < needed:
                x = np.random.randint(low, high)
                # Don't pick yourself
                if x != i:
                    arr.append(x)
            
            for x in arr:
                # We prioritize picking high-quality docs from OTHER queries
                # to serve as hard negatives
                other_doc = None
                other_item = data[x]
                
                if other_item.get('gold'):
                    other_doc = other_item['gold'][0]
                elif other_item.get('most_relevants'):
                    other_doc = other_item['most_relevants'][0]
                elif other_item.get('medium_relevants'):
                    other_doc = other_item['medium_relevants'][0]
                elif other_item.get('relevants'):
                    other_doc = other_item['relevants'][0]
                elif other_item.get('distractors'):
                    other_doc = other_item['distractors'][0]
                
                if other_doc:
                    # Append a copy of that doc so we don't link objects
                    current_distractors.append(other_doc.copy())

        final_data.append(item)
    
    return final_data

def main():
    input_file = "data/raw/eli5_org_with_qid.json"
    output_file = "data/raw/eli5_org.json"

    # 1. Load Raw
    data = load_dataset(input_file)
    
    # 2. Structure/Segregate
    segregated_data = segregate(data)
    
    # 3. Augment with negatives (Pass the segregated data here!)
    transformed_data = add_distractors(segregated_data)

    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(transformed_data, f, indent=4)

if __name__ == "__main__":
    main()