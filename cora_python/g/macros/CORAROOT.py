import os

def CORAROOT():
    """
    CORAROOT - returns the CORA root path

    Returns:
        str: The absolute path to the CORA root directory.
    """
    # Get the directory of the current file
    # __file__ is the path to the current script
    # e.g., /path/to/Translate_Cora/cora_python/g/macros/CORAROOT.py
    s = os.path.abspath(__file__)
    
    # corapath = fileparts(fileparts(fileparts(s)));
    # In Python, this is equivalent to getting the parent directory three times.
    # 1. os.path.dirname(s) -> /path/to/Translate_Cora/cora_python/g/macros
    # 2. os.path.dirname(...) -> /path/to/Translate_Cora/cora_python/g
    # 3. os.path.dirname(...) -> /path/to/Translate_Cora/cora_python
    # This seems to be one level off from the MATLAB version, which ascends from
    # .../global/macros/CORAROOT.m to the main project root.
    # The python equivalent would be .../g/macros/CORAROOT.py
    # Let's adjust for the expected structure.
    # The root should be 'Translate_Cora/'.
    # current file: Translate_Cora/cora_python/g/macros/CORAROOT.py
    # dirname 1: .../macros
    # dirname 2: .../g
    # dirname 3: .../cora_python
    # dirname 4: .../Translate_Cora/ -> this seems correct for the project root
    corapath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(s))))

    return corapath

if __name__ == '__main__':
    # Example usage:
    print(f"CORA Root Path: {CORAROOT()}") 