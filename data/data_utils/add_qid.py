import json
import os
from typing import List, Dict, Any

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
            if isinstance(data, dict):
                return [data]
            return data
        except json.JSONDecodeError as e:
            f.seek(0) # Reset file pointer to the beginning
            data = [json.loads(line) for line in f if line.strip()]
            return data

def add_qid(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    qid_data = []
    for idx, item in enumerate(data):
        temp = item.copy()
        temp["query_id"] = idx + 1
        qid_data.append(temp)
    return qid_data


def main():
    input_file = "data/raw/eli5_eval_top100_calibrated.json"
    output_file = "data/raw/eli5_org_with_qid.json"

    data = load_dataset(input_file)
    data_with_qid = add_qid(data)

    with open(output_file, "w") as f:
        json.dump(data_with_qid, f, indent=4)



if __name__ == "__main__":
    main()