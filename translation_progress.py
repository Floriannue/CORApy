import os
import argparse

def get_files(directory):#including private 
    files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        if '__pycache__' in dirnames:
            dirnames.remove('__pycache__')
        
        for filename in filenames:
            if (filename.endswith('.m') or filename.endswith('.py')) and filename != '__init__.py':
                if filename not in files:
                    files.append(filename)
    return files

def compare_files(dir1, dir2):
    if "cora_matlab" in dir2:
        return compare_files(dir2, dir1)
    
    files1 = get_files(dir1)#matlab
    files2 = get_files(dir2)#python

    missing_files = []
    for matlab_file in files1:
        matlab_file_transformed = matlab_file.lower().replace(".m", "").replace("_", "")
        translated = False
        for python_file in files2:
            python_file_transformed = python_file.lower().replace(".py", "").replace("_", "")
            if matlab_file_transformed == python_file_transformed:
                translated = True
                break
            elif matlab_file_transformed+"op" == python_file_transformed:
                translated = True
                break
        if not translated:
            missing_files.append(matlab_file)
    return missing_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translation Progress Checker - Compare MATLAB and Python directories to identify untranslated methods.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
    # Check translation progress for contSet base class
    python translation_progress.py "cora_matlab/contSet/@contSet" "cora_python/contSet/contSet"
"""
    )
    parser.add_argument('dir1', type=str, help="Matlab directory.")
    parser.add_argument('dir2', type=str, help="Python directory.")
    args = parser.parse_args()

    missing = compare_files(args.dir1, args.dir2)
    if missing:
        print(f"Python translation is missing {len(missing)} files: {missing}")
    else:
        print("All MATLAB files have been translated!")
    

