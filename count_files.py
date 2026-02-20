import os
import argparse

# (matlab_path, python_path)
MODULES = [

    # ROOT
    ("cora_matlab", "cora_python"),

    # contDynamics
    ("cora_matlab/contDynamics", "cora_python/contDynamics"),
    ("cora_matlab/contDynamics/@contDynamics", "cora_python/contDynamics/contDynamics"),
    ("cora_matlab/contDynamics/@linearARX", "cora_python/contDynamics/linearARX"),
    ("cora_matlab/contDynamics/@linParamSys", "cora_python/contDynamics/linearParamSys"),
    ("cora_matlab/contDynamics/@linearSys", "cora_python/contDynamics/linearSys"),
    ("cora_matlab/contDynamics/@nonlinearARX", "cora_python/contDynamics/nonlinearARX"),
    ("cora_matlab/contDynamics/@nonlinearSys", "cora_python/contDynamics/nonlinearSys"),
    ("cora_matlab/contDynamics/@nonlinearSysDT", "cora_python/contDynamics/nonlinearSysDT"),

    # contSet
    ("cora_matlab/contSet", "cora_python/contSet"),
    ("cora_matlab/contSet/@affine", "cora_python/contSet/affine"),
    ("cora_matlab/contSet/@capsule", "cora_python/contSet/capsule"),
    ("cora_matlab/contSet/@conPolyZono", "cora_python/contSet/conPolyZono"),
    ("cora_matlab/contSet/@contSet", "cora_python/contSet/contSet"),
    ("cora_matlab/contSet/@conZonotope", "cora_python/contSet/conZonotope"),
    ("cora_matlab/contSet/@ellipsoid", "cora_python/contSet/ellipsoid"),
    ("cora_matlab/contSet/@emptySet", "cora_python/contSet/emptySet"),
    ("cora_matlab/contSet/@fullspace", "cora_python/contSet/fullspace"),
    ("cora_matlab/contSet/@interval", "cora_python/contSet/interval"),
    ("cora_matlab/contSet/@levelSet", "cora_python/contSet/levelSet"),
    ("cora_matlab/contSet/@polygon", "cora_python/contSet/polygon"),
    ("cora_matlab/contSet/@polytope", "cora_python/contSet/polytope"),
    ("cora_matlab/contSet/@polyZonotope", "cora_python/contSet/polyZonotope"),
    ("cora_matlab/contSet/@probZonotope", "cora_python/contSet/probZonotope"),
    ("cora_matlab/contSet/@spectraShadow", "cora_python/contSet/spectraShadow"),
    ("cora_matlab/contSet/@taylm", "cora_python/contSet/taylm"),
    ("cora_matlab/contSet/@zonoBundle", "cora_python/contSet/zonoBundle"),
    ("cora_matlab/contSet/@zonotope", "cora_python/contSet/zonotope"),
    ("cora_matlab/contSet/@zoo", "cora_python/contSet/zoo"),

    # hybridDynamics
    ("cora_matlab/hybridDynamics", "cora_python/hybridDynamics"),
    ("cora_matlab/hybridDynamics/@abstractReset", "cora_python/hybridDynamics/abstractReset"),
    ("cora_matlab/hybridDynamics/@hybridAutomaton", "cora_python/hybridDynamics/hybridAutomaton"),
    ("cora_matlab/hybridDynamics/@hybridDynamics", "cora_python/hybridDynamics/hybridDynamics"),
    ("cora_matlab/hybridDynamics/@linearReset", "cora_python/hybridDynamics/linearReset"),
    ("cora_matlab/hybridDynamics/@location", "cora_python/hybridDynamics/location"),
    ("cora_matlab/hybridDynamics/@nonlinearReset", "cora_python/hybridDynamics/nonlinearReset"),
    ("cora_matlab/hybridDynamics/@transition", "cora_python/hybridDynamics/transition"),

    # nn
    ("cora_matlab/nn", "cora_python/nn"),
    ("cora_matlab/nn/layers", "cora_python/nn/layers"),
    ("cora_matlab/nn/@neuralNetwork", "cora_python/nn/neuralNetwork"),
    ("cora_matlab/nn/+nnHelper", "cora_python/nn/nnHelper"),
    ("cora_matlab/nn/optim", "cora_python/nn/optim"),
    ("cora_matlab/nn/rl", "cora_python/nn/rl"),

    # other modules
    ("cora_matlab/converter", "cora_python/converter"),
    ("cora_matlab/discrDynamics", "cora_python/discrDynamics"),
    ("cora_matlab/examples", "cora_python/examples"),
    ("cora_matlab/global", "cora_python/g"),
    ("cora_matlab/matrixSet", "cora_python/matrixSet"),
    ("cora_matlab/models", "cora_python/models"),
    ("cora_matlab/specification", "cora_python/specification"),
]


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
                matlab_path = dirpath.replace("cora_python", "cora_matlab").replace("\\g", "\\global").replace("\\nnHelper", "\\+nnHelper")
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

    #similar, compare all modules
    python count_files.py "all" "modules"

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
    elif args.dir1=="all" and args.dir2 == "modules":
        for matlab_path, python_path in MODULES:
            compare_one(matlab_path, python_path)
    else:
        compare_one(args.dir1, args.dir2)
