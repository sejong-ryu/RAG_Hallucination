import pandas as pd

file_path = "results/llama-2-7b-hf/HotpotQA-2000-2per-seed0-cons5.parquet"
new_path = "results/llama-2-7b-hf/HotpotQA-2000-2per-seed0-cons5.jsonl"
#file_path = "datasets/validation/HotpotQA.parquet"
#new_path = "datasets/validation/HotpotQA.jsonl"
df = pd.read_parquet(file_path)
df.to_json(new_path, orient="records", lines=True)