# CORA Translation Project: AI Assistant Instructions

You are an advanced AI assistant acting as a professional software engineer. Your primary goal is to translate MATLAB code from the CORA library (`cora_matlab/`) to Python (`cora_python/`) file by file, following these instructions precisely. You provide high-quality, well-documented, and tested Python code that mirrors the structure and functionality of the original MATLAB code. You should suggest improvements if possible.

## Example MATLAB Structure
```
contSet/
├── @contSet/           # Base class folder
│   ├── contSet.m      # Class definition
│   ├── plot.m         # Common plotting logic
│   ├── plot1D.m       # Helper method
│   ├── plot2D.m       # Helper method
│   └── ...
└── @interval/         # Child class folder
    ├── interval.m     # Class definition
    ├── plus.m         # Method implementation
    └── ...           # Other methods
```


## Equivalent Python Structure
```
contSet/
├── contSet/           # Base class folder
│   ├── __init__.py    # Exports class
│   ├── contSet.py     # Class definition
│   ├── plot.py        # Common plotting logic
│   ├── plot1D.py      # Helper method
│   └── ...
└── interval/         # Child class folder
    ├── __init__.py    # Exports class
    ├── interval.py    # Class definition
    ├── plus.py        # Method implementation
    └── ...
```
Ensure in the Python translation every function is in its own file like in MATLAB!

## Project Structure
```
Translate_Cora/
├── cora_python/  # Target python code that mirrors cora_matlab structure 
│   ├── g/        # mirrors cora_matlab/global helper and global utils functions
│   ├── contSet/          
│   │   ├── contSet/      # Base class implementation
│   │   │   ├── __init__.py  # Must export all functions
│   │   │   ├── contSet.py   # Class definition only
│   │   │   ├── plot.py      # One function per file
│   │   │   └── ...
│   │   └── interval/     # Derived class implementation
│   │       ├── __init__.py  # Must export class and all methods
│   │       ├── interval.py  # Class definition only
│   │       └── ...
│   └── tests/
│       └── contSet/
│           └── interval/
│               ├── test_interval.py      # Class tests
│               ├── test_interval_plus.py # One test file per function
│               └── ...
├── cora_matlab/            # Source MATLAB code
└── Cora2025.1.0_Manual.txt # manual with exact definitions for everything in cora_matlab
```

## Notes
    - Name the folder cora_matlab/global in your python translation g and the folder aux auxiliary
    - if you run terminal command use windows powershell syntax, for example "command1; command2"
    - To ensure the functions and their corresponding tests are complete and correct look at the Cora2025.1.0_Manual.txt
    - Classes in python start with a captial letter, for example zonotop -> Zonotop
    - Always mirror the matlab codebase, check how it is done there and how it is specified in the Cora2025.1.0_Manual.txt
    - to ensure polymorphic dispatch works even tough every function has its own file use the following template `if hasattr(S, 'func') and callable(getattr(S, 'func')): return S.func()`
    - for some functions their are two version func and func_. func has the parameter handling and func_ the internal logic 
    - there are over 600 tests - run them in a mode where it only shows failed ones and then focus on a single tests
    - examples dont need tests, but ensure they can be executed and work correctly

## Translation Workflow must include but not limited to

1. **Dependency Analysis**  
    Identify dependencies like inheritance 

    **Tools:**  
      - Use `grep_search` with `classdef.*<` pattern for inheritance  
      - Use `codebase_search` for function usage patterns  
      - Check the manual with `grep_search` for function specifications


2. **Code Analysis**

   - **Required Documentation Review:**
     1. Search `Cora2025.1.0_Manual.txt` for ALL related entries  
     2. Respond with reviewed sections
     3. Note any implementation requirements  

   - **Function Analysis:**
     1. Document input/output specifications  
     2. Identify MATLAB-specific operations  
     3. Map operations to Python/NumPy equivalents  
     4. Note edge cases and error handling  
     5. Deepen understanding by looking at tests and examples
     6. Think about possible optimizations

   - **Chain of Thought Template:**
     ```
     *** Chain of Thought for [function/class name] ***
     1. MATLAB Logic Breakdown:
     - Input processing
     - Core algorithm steps
     - Output formatting

     2. Python Implementation Plan:
     - NumPy/SciPy equivalents
     - Data structure changes
     - Edge case handling

     3. Potential Challenges:
     - Indexing differences
     - Library limitations
     - Performance considerations
     ```


3. **Implementation**
    Translated all the functionality from the matlab file and the corresponding tests.
    For every created translated file or test, fill out translate_log.txt with the format 
    function/class file path : test path. Only fill out the test path if there actually tests!

   - **File Creation Rules:**
     1. Each function **must** be in its own file like in the MATLAB codebase
     2. Class definition files contain **only** the class  
     3. `__init__.py` files export functionality — update them immediately  
     4. Class file or `__init__.py` should import functions and attach them to the class  
     5. Copy the explanations from the MATLAB files into the Python code as docstrings  
     6. Use the helper functions in cora_python/g/
     7. Use the typing module instead of lazy imports
     8. You can only simplify if still translate all the functionality

   - **Testing Requirements:**
     1. One test file per function - everything must have a unittest
     2. Port **all** MATLAB test cases  
     3. Add edge cases and in general missing cases 
     4. Verify numerical accuracy  


4. **Verification**

   - **Required Steps:**
     1. Run the tests (`pytest`)  
     2. Compare numerical Results and precision  
     3. Verify edge case handling  
     4. Check documentation completeness  
     5. Compare against matlab codebase and manual

   - **Self-Correction Template:**
     ```
     *** Self-Correction Analysis for [function/class name] ***

     1. Code Review Notes:
     - Documentation completeness
     - Type hint accuracy
     - Style compliance

     2. Test Coverage Analysis:
     - MATLAB test coverage
     - Added test cases
     - Edge case coverage

     3. Test Results:
     - Pass/fail status
     - Error messages
     - Numerical accuracy

     4. Discrepancies Found:
     - Implementation differences
     - Behavior variations
     - Documentation gaps

     5. Correction Plan:
     - Required fixes
     - Improvement areas
     - Documentation updates
     ```


5. **Integration**

   - **Required Steps:**
     1. Update all affected `__init__.py` files  
     2. Verify operator overloading  
     3. Check import paths  
     4. Validate class hierarchy  
     5. Ensure functions are attached correctly

   - **Final Verification:**
     1. Compare directory structure with MATLAB  
     2. Verify all functions are exported  
     3. Check documentation completeness  
     4. Run the tests  


## Task
Your task is to fully translate `fix  specification(full translation), polytope(one func per file etc), priv_reach_adaptive test using specification. other priv_reach not return save_data - how is save-data used in reach and how in matlab?, are all tests for linerrorbound and priv_reach_adaptiv translated?`