import re
from collections import defaultdict
import sys

"""
git diff --numstat ba600c7c109cc1e9d5533ab5393936dc342a47f5 > t.txt
"""

def parse_git_stat(file_path):
    folder_changes = defaultdict(lambda: {"add": 0, "changed": 0})

    pattern = re.compile(r'^\s*(\d+|-)\s+(\d+|-)\s+(.+)$')

    with open(file_path, 'r', encoding='utf-16') as f:
        for line in f:
            match = pattern.match(line)
            if not match:
                continue

            insertions, deletions, filepath = match.groups()

            if insertions == '-' or deletions == '-':
                continue

            insertions = int(insertions)
            deletions = int(deletions)

            # classify lines cleanly
            added = max(insertions - deletions, 0)
            changed = min(insertions, deletions)

            parts = filepath.split('/')

            # accumulate for ALL parent directories
            for i in range(1, len(parts)):
                folder = "/".join(parts[:i])
                folder_changes[folder]["add"] += added
                folder_changes[folder]["changed"] += changed

    return folder_changes



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_git_diff.py <git_numstat_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    results = parse_git_stat(file_path)

    print("\nChanged lines per folder:\n")
    for folder, d in sorted(
        results.items(),
        key=lambda x: x[1]['add'] + x[1]['changed'],
        reverse=True
    ):
        print(f"{folder}: {d['add']} added, {d['changed']} changed")
