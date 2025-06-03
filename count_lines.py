import os

def count_lines_of_code(directory):
    total_lines = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.m'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('%')]
                    total_lines += len(lines)
    return total_lines

path = os.path.join(os.getcwd(), 'cora_matlab')
print(count_lines_of_code(path))