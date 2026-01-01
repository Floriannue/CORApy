import os

def CORAROOT():
    """
    CORAROOT - returns the CORA root path (cora_python directory)

    Returns:
        str: The absolute path to the cora_python directory.
    """
    # Get the directory of the current file
    # __file__ is the path to the current script
    # e.g., /path/to/Translate_Cora/cora_python/g/macros/CORAROOT.py
    s = os.path.abspath(__file__)
    
    # MATLAB: corapath = fileparts(fileparts(fileparts(s)));
    # MATLAB file: cora_matlab/global/macros/CORAROOT.m
    # dirname 1: .../macros
    # dirname 2: .../global
    # dirname 3: .../cora_matlab
    # 
    # Python equivalent: cora_python/g/macros/CORAROOT.py
    # dirname 1: .../macros
    # dirname 2: .../g
    # dirname 3: .../cora_python
    corapath = os.path.dirname(os.path.dirname(os.path.dirname(s)))

    return corapath

if __name__ == '__main__':
    # Example usage:
    print(f"CORA Root Path: {CORAROOT()}") 