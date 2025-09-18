from __future__ import print_function
from typing import Dict, List
import json
import os


def read_json(file_path):
    """Read and return the content of a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}.")
        return []


def save_json_file(data, file_path):
    """Save data to a JSON file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")


def load_jsonl_dataset(jsonl_file: str) -> List[Dict]:
    """Đọc toàn bộ file .jsonl, mỗi dòng là 1 dict."""
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                data.append(record)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON parse lỗi ở dòng {line_idx}: {e}")
    print(f"✅ Loaded {len(data)} samples từ {jsonl_file}")
    return data


def batchify(data, batch_size: int):
    """Chia data thành batch."""
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def save_jsonl_file(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def make_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir
