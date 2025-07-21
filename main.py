from agent.workflow import competitive_score
import sys
import os
from utils.input import extract_targets_from_file

def main():
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        file_path = sys.argv[1]
        targets = extract_targets_from_file(file_path)
        if not targets:
            print("No targets found in the file.")
            return
        print("Targets found in file:")
        for t in targets:
            print(f"- {t}")
        print("-" * 40)
        for target in targets:
            result = competitive_score(target)
            print(f"Target: {target}")
            print(result["competitive_analysis"])
            print("-" * 40)
    else:
        target = input("Enter the molecule target: ")
        if not target.strip():
            print("No target provided.")
            return
        print(f"Target: {target}")
        result = competitive_score(target)
        print(result["competitive_analysis"])

if __name__ == "__main__":
    main()