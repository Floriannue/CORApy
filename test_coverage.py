import os
import argparse

def get_files(directory):#including private 
    files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        if '__pycache__' in dirnames:
            dirnames.remove('__pycache__')
        
        for filename in filenames:
            if (filename.endswith('.py')) and filename != '__init__.py':
                if filename not in files:
                    files.append(filename)
    return files

def compare_files(dir, test_dir):
    if "tests" in dir:
        return compare_files(test_dir, dir)

    implementation_files = get_files(dir)
    test_files = get_files(test_dir) or []

    if implementation_files is None:
        return []

    untested_methods = []
    for file in implementation_files:
        file_transformed = file.lower().replace('_op', '').replace('_', '')
        tested = False
        for test in test_files:
            test_transformed = test.lower().replace('_op', '').replace('_', '').replace("constructor","")
            # Direct match: function.py -> test_function.py
            if f"test{file_transformed}" == test_transformed:
                tested = True
                break
            # Class-specific test: method.py -> test_<class>_method.py
            elif test_transformed.endswith(f"{file_transformed}"):
                tested = True
                break
            # Exact filename match (should not happend)
            elif file_transformed == test_transformed:
                tested = True 
                break
                
        if not tested:
            untested_methods.append(file)
            
    return untested_methods

def calculate_coverage_percentage(dir, test_dir):
    """
    Calculate test coverage percentage.
    
    Returns:
        Tuple of (coverage_percentage, total_methods, tested_methods)
    """
    implementation_files = get_files(dir) or []
    untested = compare_files(dir, test_dir)
    
    total_methods = len(implementation_files)
    tested_methods = total_methods - len(untested)
    coverage_percentage = (tested_methods / total_methods * 100) if total_methods > 0 else 0
    
    return coverage_percentage, total_methods, tested_methods

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Coverage Checker - Identify Python implementation files that lack corresponding unit tests.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
    # Check test coverage for interval class
    python test_coverage.py "cora_python/contSet/interval" "cora_python/tests/contSet/interval"
"""
    )
    parser.add_argument('dir1', type=str, help="Python directory.")
    parser.add_argument('dir2', type=str, help="Tests directory.")
    args = parser.parse_args()

    missing = compare_files(args.dir1, args.dir2)
    coverage_percent, total_methods, tested_methods = calculate_coverage_percentage(args.dir1, args.dir2)
    if missing:
        print(f"Python translation has {coverage_percent}% coverage: {tested_methods}/{total_methods} methods tested")
        print(f"Python translation is missing {len(missing)} testfiles: {missing}")
    else:
        print(f"All {args.dir1} files are tested! Python translation has {coverage_percent}% coverage: {tested_methods}/{total_methods} methods tested")
    

