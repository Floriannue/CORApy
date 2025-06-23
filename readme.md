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

> Ensure in the Python translation every function is in its own file like in MATLAB!

## Project Structure
```
Translate_Cora/
├── cora_python/  # Target python code that mirrors cora_matlab structure 
│   ├── g/        # mirrors cora_matlab/global helper and utility functions
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
- The methode object.display() should return the string and not print it since display also provides the string for __str__
- Dont catch warnings
- Use the objects methodes instead of importing them since class methodes are attached so use 
  ```python
  object.function()
  ```
  instead of 
    ```python
  import function
  function(object)
  ```
- Always provide a full translation and no simplified verison that are missing features
- Treat everything as modules. For example, to execute `cora_python/folder/func.py`, use:  
  ```powershell
  python -m cora_python.folder.func
  ```
- Name the folder `cora_matlab/global` as `g` and `aux` as `auxiliary` in the Python translation.
- If you run a terminal command, use Windows PowerShell syntax, e.g.,  
  ```powershell
  command1; command2 | Select-String "string"
  ```
- To ensure the functions and their corresponding tests are complete and correct, refer to `Cora2025.1.0_Manual.txt`.
- Classes in Python start with a capital letter. For example, `zonotop` → `Zonotop`.
- Always mirror the MATLAB codebase and verify against the manual.
- Use the following two polymorphic dispatch templates depending on the situation:
  ```python
  #1
  def func():
    # Check if subclass has overridden func method
    if type(self).func is not parent.func:
        return type(self).func(self)
    else:
        from .func import func
        return func(self)

  #2
  def otherFunc():
    if hasattr(self, 'func') and callable(getattr(self, 'func')):
      return self.func(point)
  ```
- For functions with `func` and `func_`, `func` handles parameters, and `func_` is the internal logic.
- There are over 600 tests. Run them in a mode that shows **only failed** ones, then debug one by one.
- Examples do not need tests, but they **must** execute correctly.
- Use `d x n` vertices format:  
  ```python
  #Ensure the matric and vector format work togther
  np.array([[0, 1, 0], [0, 0, 1]])  # == 2×3 matrix
  np.array([1, 0])  #vector
  ```
- For functions that plot, save the output as PNG and verify visually.
- To capture large function output:
  ```powershell
  function-call > output.txt
  ```


## Translation Workflow must include but not limited to

### 1. **Dependency Analysis** 
Identify dependencies like inheritance. 

#### Tools:  
- Use `grep_search` with `classdef.*<` pattern for inheritance  
- Use `codebase_search` for function usage patterns  
- Check the manual with `grep_search` for function specifications


### 2. **Code Analysis**

#### Required Documentation Review:
1. Search `Cora2025.1.0_Manual.txt` for ALL related entries  
2. Respond with reviewed sections
3. Note any implementation requirements  

#### Function Analysis:
1. Document input/output specifications  
2. Identify MATLAB-specific operations  
3. Map operations to Python/NumPy equivalents  
4. Note edge cases and error handling  
5. Deepen understanding by looking at tests and examples
6. Think about possible optimizations

#### Chain of Thought Template:
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


### 3. **Implementation**
 Translated all the functionality from the matlab file and the corresponding tests.
 For every created translated file or test, fill out translate_log.txt with the format 
 function/class file path : test path. Only fill out the test path if there are actually tests!

#### File Creation Rules:
1. Each function **must** be in its own file like in the MATLAB codebase, except if there are multiple functions in one file in matlab they all should also be in one file in python! In doubt copy the matlab structure
2. Class definition files contain **only** the class  
3. `__init__.py` files export methods and attach them to the class if possible — update them immediately 
4. Copy the explanations from the MATLAB files into the Python code as docstrings  
5. Use the helper functions in cora_python/g/
6. To prevent circular imports use the typing module for class imports. Only use lazy function imports as last resort
7. Optimize only if the functionality is fully preserved
8. Do not implement silent fails

#### Testing Requirements:
1. One test file per function - everything must have a unit test
2. Port **all** MATLAB test cases  
3. Add edge cases and in general missing cases 
4. Verify numerical accuracy  


### 4. **Verification**

#### Required Steps:
1. Run the tests (`pytest`)  
2. Compare numerical Results and precision  
3. Verify edge case handling  
4. Check documentation completeness  
5. Translated code or test can be wrong therefore compare against matlab codebase and manual

#### Self-Correction Template:
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


### 5. **Integration**

#### Required Steps:
1. Update all affected `__init__.py` files  
2. Verify operator overloading  
3. Check import paths  
4. Validate class hierarchy  
5. Ensure functions are attached correctly

#### Final Verification:
1. Compare directory structure with MATLAB  
2. Verify all functions are exported  
3. Check documentation completeness  
4. Run the tests  


## Task
Your task is to `fix all errors and translate missing tests for interval, zonotope, polytope, capsule, ellipsoid, emptySet, contDynamics, linearSys, g`. if there are no test files to translate for a function or class create appropriate ones by looking at the manual. also add missing test cases. Also change all the lazy imports (imports inside functions to prevent circular imports) to imports with the typing module!
