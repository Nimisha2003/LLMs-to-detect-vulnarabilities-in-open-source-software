import json
import os
import pandas as pd

OUT_JSONL = "merged_dataset.jsonl"

def write_jsonl(records, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_bigvul(path):
    df = pd.read_parquet(path, engine="pyarrow")
    records = []
    for _, row in df.iterrows():
        code = row.get("func_before") or ""   # vulnerable code
        label = int(row.get("vul") or 0)      # 1 vulnerable / 0 safe
        language = row.get("lang") or "C"
        cwe = row.get("CWE ID") or ""
        if isinstance(code, str) and code.strip():
            records.append({
                "code": code,
                "label": label,
                "language": str(language),
                "source": "BigVul",
                "cwe": str(cwe)
            })
    return records

def load_devign(path):
    df = pd.read_parquet(path, engine="pyarrow")
    records = []
    for _, row in df.iterrows():
        code = row.get("func") or row.get("func_clean") or row.get("normalized_func") or ""
        
        # Fix: handle label arrays/lists
        label_val = row.get("target") if "target" in row else row.get("label")
        if isinstance(label_val, (list, tuple)):
            label = int(label_val[0])
        elif hasattr(label_val, "item"):  # numpy scalar
            label = int(label_val.item())
        else:
            label = int(label_val or 0)

        language = row.get("project") or "C"
        if isinstance(code, str) and code.strip():
            records.append({
                "code": code,
                "label": label,
                "language": str(language),
                "source": "Devign",
                "cwe": ""   # Devign doesn’t always have CWE
            })
    return records

def main():
    records = []

    # BigVul files
    bigvul_dir = r"C:\Users\nimis\Downloads\first phase\llm_vuln_detector 2\data\bigvul\data"
    for fname in os.listdir(bigvul_dir):
        if fname.endswith(".parquet"):
            print("Loading BigVul:", fname)
            records.extend(load_bigvul(os.path.join(bigvul_dir, fname)))

    # Devign files
    devign_dir = r"C:\Users\nimis\Downloads\first phase\llm_vuln_detector 2\data\devign\data"
    for fname in os.listdir(devign_dir):
        if fname.endswith(".parquet"):
            print("Loading Devign:", fname)
            records.extend(load_devign(os.path.join(devign_dir, fname)))

    print(f"Total records merged: {len(records)}")
    write_jsonl(records, OUT_JSONL)
    print("Saved:", OUT_JSONL)

if __name__ == "__main__":
    main()