import os
import re

def fix_imports_in_file(file_path):
    """
    Reads a Python file and fixes incorrect contSet import statements.
    """
    content = None
    original_encoding = 'utf-8'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            original_encoding = 'latin-1'
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path} with any encoding: {e}")
            return

    # This regex specifically targets incorrect contSet imports, e.g.,
    # from cora_python.contSet.zonotope.zonotope import Zonotope
    # and rewrites it to the correct form:
    # from cora_python.contSet import Zonotope
    new_content, count = re.subn(
        r"from cora_python\.contSet\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+ import ([a-zA-Z0-9_]+)",
        r"from cora_python.contSet import \1",
        content
    )

    if count > 0:
        print(f"Fixed {count} import(s) in: {file_path}")
        #with open(file_path, 'w', encoding=original_encoding) as f:
            #f.write(new_content)

def main():
    """
    Walks through the cora_python directory and fixes imports in all .py files.
    """
    start_dir = 'cora_python'
    for root, _, files in os.walk(start_dir):
        for file in files:
            if file.endswith('.py'):
                fix_imports_in_file(os.path.join(root, file))

if __name__ == "__main__":
    main()
    print("Import fixing script finished.") 