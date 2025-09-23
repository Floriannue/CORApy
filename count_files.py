import os
import argparse

def count_files_in_directory(directory):
    """
    Counts the number of files in a given directory.
    Excludes __pycache__ and simliar stuff
    """
    if directory == None:
        return None
    
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        return None

    file_count = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        if '__pycache__' in dirnames:
            dirnames.remove('__pycache__')
        
        for filename in list(filenames): #list(filenames) should not be a new object not affected by remove
            for exclude in ['.md', '.json', '.ds_store', '.txt', '__init__.py']:
                if exclude in filename.lower():
                    filenames.remove(filename)
                    break

        file_count += len(filenames)

    return file_count

def compare_one(dir1, dir2, filter=False):
    count1 = count_files_in_directory(dir1)
    count2 = count_files_in_directory(dir2)
    difference = -1
    if count1 and count2:
        difference = abs(count1 - count2)

    if filter and difference <= 0:
        return #filter out flolders with equal file count
    
    if count1:print(f"Total relevant files in '{dir1}': {count1}")
    if count2:print(f"Total relevant files in '{dir2}': {count2}")
    if difference != -1:print(f"The difference is {difference}. {count2/count1*100}% translated.") 

def compare_all():
    for dirpath, dirnames, filenames in os.walk("cora_python"):#path, contained folders, contained files
        #no example, tests, only file folder
        if not "example" in dirpath and not "tests" in dirpath and not"__pycache__" in dirpath:
            if filenames != [] and filenames != ['__init__.py']:
                matlab_path = dirpath.replace("cora_python", "cora_matlab").replace("\\g", "\\global")
                if not os.path.exists(matlab_path):
                    temp_path = matlab_path.split("\\")
                    if 'private' in dirpath:
                        temp_path[-2] = "@" + temp_path[-2]
                    else:  
                        temp_path[-1] = "@" + temp_path[-1]
                    matlab_path = "\\".join(temp_path)
                compare_one(matlab_path, dirpath, filter=True)
                print("----------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count and compare relevant files in one or two directories, excluding specified non-code files",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
    #compare all relevant python folders against matlab
    python count_files.py

    # Count translatable files in the MATLAB contSet directory
    python count_files.py "cora_matlab/contSet/@contSet"

    # Compare file counts between MATLAB and Python contSet directories
    python count_files.py "cora_matlab/contSet/@contSet" "cora_python/contSet/contSet"
"""
    )
    parser.add_argument('dir1', type=str, help="First directory (optional).", nargs='?', default=None)
    parser.add_argument('dir2', type=str, help="Second directory to compare against (optional).", nargs='?', default=None)
    args = parser.parse_args()

    if not args.dir1 and not args.dir2:
        compare_all()
    else:
        compare_one(args.dir1, args.dir2)
