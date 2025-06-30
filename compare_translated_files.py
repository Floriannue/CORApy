import os
import argparse

def get_files(directory):#including private 
    files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        if '__pycache__' in dirnames:
            dirnames.remove('__pycache__')

        if dirnames != [] and dirnames != ['private']:
            print(f"Error: folder '{directory}' has unexpected subfolders.")
            return None
        
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
        matlab_file = matlab_file.replace(".m", "")
        translated = False
        for python_file in files2:
            python_file = python_file.replace(".py", "")
            if matlab_file == python_file:
                translated = True
                break
            elif matlab_file+"_op" == python_file:
                translated = True
                break
            elif any(char.isupper() for char in matlab_file):
                #find upper char and replace it with _char since python - matlab naming convention
                for char in matlab_file:
                    if char.isupper():
                        mfile1 = matlab_file.replace(char, "_"+char)
                        mfile2 = matlab_file.replace(char, "_"+char.lower())
                        break
                if mfile1 == python_file or mfile2 == python_file:
                    translated = True
                    break
        if not translated:
            missing_files.append(matlab_file)
    return missing_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare available matlab/python files in one or two directories, excluding specified non-code files.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
    # Compare file Names between MATLAB and Python contSet directories
    python compare_translated_files.py "cora_matlab/contSet/@contSet" "cora_python/contSet/contSet"
"""
    )
    parser.add_argument('dir1', type=str, help="Matlab directory.")
    parser.add_argument('dir2', type=str, help="Python directory.")
    args = parser.parse_args()

    missing = compare_files(args.dir1, args.dir2)
    if missing:
        print(f"Python translation is missing {len(missing)} files:")
        for file in missing:
            print(f"  - {file}")
    else:
        print("All MATLAB files have been translated!")
    

