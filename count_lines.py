import os
import argparse

#exclude comments and empty lines
def count_file(file_path, ignore_python_headers=False):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if ignore_python_headers:
            for i in range(len(lines)):
                if lines[i].startswith('def ') or lines[i].startswith('class '):
                    lines = lines[i:]
                    break
        lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('%') and not line.strip().startswith('#')]
        return len(lines)

def count_lines_of_code(directory, ignore_python_headers=False):
    total_lines = 0
    if os.path.isfile(directory):
        total_lines = count_file(directory, ignore_python_headers)
    else:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if (file.endswith('.m') or file.endswith('.py')) and not file == '__init__.py':
                    file_path = os.path.join(root, file)
                    total_lines += count_file(file_path, ignore_python_headers)
    return total_lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count and compare matlab/python code lines in one or two directories/files, excluding specified non-code files. ignore python headers when comparing.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
    # Count one directory/file
    python count_lines.py "cora_matlab/contSet/@contSet"
    python count_lines.py count_lines.py

    # Compare file counts between MATLAB and Python contSet directories
    python count_lines.py "cora_matlab/contSet/@contSet" "cora_python/contSet/contSet"
"""
    )
    parser.add_argument('dir1', type=str, help="First directory/file (required).")
    parser.add_argument('dir2', type=str, help="Second directory/file to compare against (optional).", nargs='?', default=None)
    args = parser.parse_args()

    count1 = None
    count2 = None

    if args.dir1 and args.dir2:
        count1 = count_lines_of_code(args.dir1, ignore_python_headers=True)
        count2 = count_lines_of_code(args.dir2, ignore_python_headers=True)
    elif args.dir1:
        count1 = count_lines_of_code(args.dir1)
    elif args.dir2:
        count2 = count_lines_of_code(args.dir2)

    if count1:
        print(f"The total lines of code in {args.dir1} is {count1}.")
    if count2:
        print(f"The total lines of code in {args.dir2} is {count2}.")
    if count1 and count2:
        print(f"The difference is {abs(count1 - count2)}.")
