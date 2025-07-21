import os
import json
def extract_targets_from_file(file_path):
    """
    Reads a CSV or .txt or .json file and extracts the target(s).
    Assumes each line or row contains a target string.
    Returns a list of targets.
    """
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return []

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    targets = []

    if ext == ".csv":
        with open(file_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            target_idx = None
            # Try to find the column index for 'target' or 'molecular_target'
            for i, col in enumerate(header):
                if col.lower() == "target":
                    target_idx = i
                    break
                elif col.lower() == "molecular_target":
                    target_idx = i
                    break
            for line in f:
                row = line.strip().split(",")
                if target_idx is not None and len(row) > target_idx and row[target_idx]:
                    targets.append(row[target_idx])
                elif row and row[0]:
                    targets.append(row[0])
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # Support both comma-separated and line-by-line entries
                entries = [t.strip() for t in line.strip().split(",") if t.strip()]
                targets.extend(entries)
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                if isinstance(item, dict):
                    if "target" in item:
                        targets.append(item["target"])
                    elif "molecular_target" in item:
                        targets.append(item["molecular_target"])
                    else:
                        # fallback: try to use the whole dict as a string
                        targets.append(str(item))
                else:
                    targets.append(item)
    else:
        print("Unsupported file type. Please provide a .csv or .txt file.")
        return []

    return targets

def process_targets(targets, competitive_score_func):
    """
    Loops over the list of targets and calls the provided competitive_score_func(target).
    Returns a list of (target, result) tuples.
    """
    results = []
    for target in targets:
        try:
            result = competitive_score_func(target)
            results.append((target, result))
            print(f"Target: {target} -> Result: {result}")
        except Exception as e:
            print(f"Error processing target '{target}': {e}")
    return results
