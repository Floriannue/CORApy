# CORA Translation Project: AI Assistant Instructions

You are an advanced AI assistant who is acting as a professional software engineer. Your primary goal is to translate MATLAB code from the CORA library (`cora_matlab/`) to Python (`cora_python/`) file by file, following these instructions precisely. You provide high-quality, well-documented, and tested Python code that mirrors the structure and functionality of the original MATLAB code. You should suggest improvements if possible.

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
│   │   │   ├── __init__.py  # Must export class and attache/overload all methods
│   │   │   ├── contSet.py   # Constructor only
│   │   │   ├── plot.py      # One class methode per file
│   │   │   └── ...
│   │   └── interval/     # Derived class implementation
│   │       ├── __init__.py  # Must export class and attache/overload all methods
│   │       ├── interval.py  # Constructor only
│   │       └── ...
│   └── tests/
│       └── contSet/
│           └── interval/
│               ├── test_interval.py      # Constructor tests
│               ├── test_interval_plus.py # One test file per function
│               └── ...
├── cora_matlab/            # Source MATLAB code
└── Cora2025.1.0_Manual.txt # manual with exact definitions for everything in cora_matlab
```


## Notes
- `*` operator (`__mul__`) = (times) element-wise multiplication (like MATLAB's `.*`) and `@` operator (`__matmul__`) = (mtimes) matrix multiplication (like MATLAB's `*`)
- methods that have the same name as reserved keywords in Python get the appendix _op, for example, and -> and_op
- The method object.display() should return the string and not print it since display also provides the string for `__str__`
- Dont catch warnings
- **Never** import methods as standalone functions. All methods are attached to classes in `__init__.py`.
- **Rule: Class methods must be called on an object instance, never imported directly.**  
  Functions that act as class methods (e.g., `plus.py`) are dynamically attached to their corresponding class (e.g., `Interval`) inside the `__init__.py` file. 
  This pattern is essential to mirror MATLAB's `@class` folder behavior, where any function file in a class folder is automatically a method. 
  And since in our translation we keep all the methodes also in there own seperate file, you must never import these method files and call them as if they were standalone functions.

  **WRONG:** Importing and calling the function directly.
  ```python
  # This treats the method like a standalone utility function, which is incorrect.
  from .other_methode import other_methode # <--- WRONG IMPORT
  
  # Assume 'obj_A' is an instance of the class
  result = other_methode(obj_A, b) # <--- WRONG CALL
  ```

  **CORRECT:** Calling the function as a method on the object instance.
  The `__init__.py` handles the attachment, so no import is needed in the file where you are using the object. The method is already part of the class.
  ```python
  # Assume 'obj_A' is an instance of the class.
  # The 'other_methode' is already part of its class definition via __init__.py.
  result = obj_A.other_methode(b) # <--- CORRECT CALL
  ```
- if methodes need to import the class they are part of so should it be done at the top of the file
- Always provide a full complete translation and no simplified version that is missing features or has cheap workarounds
- Treat everything as modules. For example, to execute `cora_python/folder/func.py`, use:  
 ```powershell
  python -m cora_python.folder.func
 ```
- Name the folder `cora_matlab/global` as `g` and `aux` as `auxiliary` in the Python translation.
- If you run a terminal command, use Windows PowerShell syntax, e.g.,  
 ```powershell
  command1; command2 | Select-String "string"
 ```
 also use 2>&1. if you cant get the output of a terminal command or it has been moved to background use > terminal_output.txt and read the file
  ```powershell
  command > output.txt
 ```
- To ensure the functions and their corresponding tests are complete and correct, refer to `Cora2025.1.0_Manual.txt`.
- Classes in Python start with a capital letter. For example, `zonotop` → `Zonotop`. Try to make the translation as pythonic as possible but still keeping in a full translation.
- Always mirror the MATLAB codebase and verify against it and the manual. In rare cases the matlab codebase can be wrong, in this case look at the manual and provide all information to the user!
- Use the following two polymorphic dispatch templates depending on the situation:
 ```python
  def func():
    # Check if subclass has overridden func method
    if type(self).func is not parent.func:
        return type(self).func(self)
 ```
 ```python
  def otherFunc():
    if hasattr(self, 'func') and callable(getattr(self, 'func')):
      return self.func(point)
 ```
- For functions with `func` and `func_`: `func` is the public interface with validation and error handling (func mainly exists in the parent class). func then calls func_ (polymorphism). `func_` is the raw implementation for internal use, cross-class calls, and performance-critical paths (overwritten in the child class).
- There are over 1000 tests. Use the pytests parameters wisely to not overwhelm you with to much output:
with -x it stops after the first failed tests, 
run a specific test file with -v for more output, 
use --lf to run only tests that failed last time, 
--tb=no reduces traceback if you run a lot oft tests for an overview, 
-q for less output, good for an overview.

- Examples do not need tests, but they **must** execute correctly.
- Use `d x n` vertices format:  
 ```python
  #Ensure the matrix and vector format work together
  np.array([[0, 1, 0], [0, 0, 1]])  # == 2×3 matrix
  np.array([1, 0])  #vector
 ```
- For functions that plot, save the output as PNG and verify visually.
- if there is an error find the source and compare against matlab. Do not use any cheap workarounds or simplifications! Always find and adress the route cause.
- Read the folder to see which files are there and read the files before you try to change them


## Translation Workflow must include but not limited to

### 0. **Check what is already done** 

### 1. **Dependency Analysis** 
Identify dependencies like inheritance. 
This workflow must also be applied to dependencies you translated - translate the correspond test directly after you translated the file!

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
1. Each function **must** be in its own file like in the MATLAB codebase, except if there are multiple functions in one file in matlab they all should also be in one file in Python! In doubt copy the matlab structure
2. Class definition files contain **only** the class  
3. `__init__.py` attaches methods to the class and exports the class, functions
4. Copy the explanations from the MATLAB files into the Python code as docstrings  
5. Use the helper functions in cora_python/g/
6. you nearly never need lazy imports since the importing is in `__init__.py` and not `class.py`
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
5. Translated code or test can be wrong therefore compare against the matlab codebase and manual
6. if matlab is installed, create an example for the matlab source and your python translation so you can compare the results directly

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
Your task is to translate missing `contSet methodes` and any associated missing tests (and create addtional test cases if the matlab ones are not comprehensiv)
