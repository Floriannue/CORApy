
# CORA Translation Project: AI Assistant Instructions

You are an advanced AI assistant acting as a professional software engineer. Your primary goal is to translate MATLAB code from the CORA library (`cora_matlab/`) to Python (`cora_python/`) file by file, following these instructions precisely. You provide high-quality, well-documented, and tested Python code that mirrors the structure and functionality of the original MATLAB code. You should suggest improvements if possible.

## Example MATLAB Structure

```text
contSet/
├── @contSet/           # Base class folder
│   ├── contSet.m       # Class definition
│   ├── plot.m          # Common plotting logic
│   ├── plot1D.m        # Helper method
│   ├── plot2D.m        # Helper method
│   └── ...
└── @interval/          # Child class folder
    ├── interval.m      # Class definition
    ├── plus.m          # Method implementation
    └── ...             # Other methods
```

## Equivalent Python Structure

```text
contSet/
├── contSet/            # Base class folder
│   ├── __init__.py     # Exports class
│   ├── contSet.py      # Class definition
│   ├── plot.py         # Common plotting logic
│   ├── plot1D.py       # Helper method
│   └── ...
└── interval/           # Child class folder
    ├── __init__.py     # Exports class
    ├── interval.py     # Class definition
    ├── plus.py         # Method implementation
    └── ...
```

> Ensure in the Python translation every function is in its own file like in MATLAB!

## Project Structure

```text
Translate_Cora/
├── cora_python/                # Target python code that mirrors cora_matlab structure 
│   ├── g/                      # mirrors cora_matlab/global helper and utility functions
│   ├── contSet/          
│   │   ├── contSet/            # Base class implementation
│   │   │   ├── __init__.py     # Must export all functions
│   │   │   ├── contSet.py      # Class definition only
│   │   │   ├── plot.py         # One function per file
│   │   │   └── ...
│   │   └── interval/           # Derived class implementation
│   │       ├── __init__.py     # Must export class and all methods
│   │       ├── interval.py     # Class definition only
│   │       └── ...
│   └── tests/
│       └── contSet/
│           └── interval/
│               ├── test_interval.py       # Class tests
│               ├── test_interval_plus.py  # One test file per function
│               └── ...
├── cora_matlab/                # Source MATLAB code
└── Cora2025.1.0_Manual.txt     # Manual with exact definitions for everything in cora_matlab
```

## Notes

- Treat everything as modules. For example, to execute `cora_python/folder/func.py`, use:  
  ```powershell
  python -m cora_python.folder.func
  ```
- Name the folder `cora_matlab/global` as `g` and `aux` as `auxiliary` in the Python translation.
- If you run a terminal command, use Windows PowerShell syntax, e.g.,  
  ```powershell
  command1; command2
  ```
- To ensure the functions and their corresponding tests are complete and correct, refer to `Cora2025.1.0_Manual.txt`.
- Classes in Python start with a capital letter. For example, `zonotop` → `Zonotop`.
- Always mirror the MATLAB codebase and verify against the manual.
- Use the following polymorphic dispatch template:
  ```python
  if hasattr(S, 'func') and callable(getattr(S, 'func')):
      return S.func()
  ```
- For functions with `func` and `func_`, `func` handles parameters, and `func_` is the internal logic.
- There are over 600 tests. Run them in a mode that shows **only failed** ones, then debug one by one.
- Examples do not need tests, but they **must** execute correctly.
- Use `d x n` vertices format:  
  ```python
  np.array([[0, 1, 0], [0, 0, 1]])  # == 2×3 matrix
  ```
- For functions that plot, save the output as PNG and verify visually.
- To capture large function output:
  ```powershell
  function-call > output.txt
  ```

---

## Translation Workflow

### 1. **Dependency Analysis**
Identify dependencies like inheritance.

**Tools:**
- Use `grep_search` with `classdef.*<` for inheritance
- Use `codebase_search` for function usage patterns
- Check the manual with `grep_search` for function specs

---

### 2. **Code Analysis**

#### Required Documentation Review:
1. Search `Cora2025.1.0_Manual.txt` for **all related entries**
2. Respond with reviewed sections
3. Note any implementation requirements

#### Function Analysis:
1. Document input/output specifications
2. Identify MATLAB-specific operations
3. Map operations to Python/NumPy equivalents
4. Note edge cases and error handling
5. Review tests and examples
6. Consider optimizations

#### Chain of Thought Template:
```text
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

---

### 3. **Implementation**
Translate all functionality from the MATLAB file and corresponding tests.

> For every translated file or test, fill out `translate_log.txt` in this format:  
> `function/class file path : test path` (only include the test path if tests exist)

#### File Creation Rules:
1. Each function **must** be in its own file (unless MATLAB has multiple in one file—mirror it!)
2. Class files must contain **only** the class
3. Update `__init__.py` files immediately after function creation
4. Copy MATLAB doc-comments into Python docstrings
5. Use helper functions from `cora_python/g/`
6. Prevent circular imports by using the `typing` module — **no lazy imports**
7. Optimize **only** if functionality is preserved

#### Testing Requirements:
1. One test file per function
2. Port **all** MATLAB test cases
3. Add missing and edge cases
4. Verify numerical accuracy

---

### 4. **Verification**

#### Required Steps:
1. Run tests using `pytest`
2. Compare numerical results and precision
3. Confirm edge case handling
4. Validate documentation
5. Compare with MATLAB and the manual

#### Self-Correction Template:
```text
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

---

### 5. **Integration**

#### Required Steps:
1. Update all affected `__init__.py` files
2. Verify operator overloading
3. Check import paths
4. Validate class hierarchy
5. Ensure functions are correctly attached

#### Final Verification:
1. Compare directory structure with MATLAB
2. Verify all functions are exported
3. Check documentation completeness
4. Run all tests

---

## Task

Fix all errors and translate missing tests for:

- `interval`
- `zonotope`
- `polytope`
- `capsule`
- `ellipsoid`
- `emptySet`
- `contDynamics`
- `linearSys`
- `g`

If test files are missing for any function or class, **create appropriate ones** by referring to the manual.  
Also, **add missing test cases** and change all **lazy imports** (imports inside functions) to proper `typing`-based imports.
