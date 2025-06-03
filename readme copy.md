# CORA Translation Project: MATLAB to Python

## Project Overview
This project aims to translate the CORA toolbox from MATLAB to Python while maintaining functionality and performance.

## Directory Structure
- `cora_matlab/` - Original MATLAB implementation
- `cora_python/` - Python implementation (in progress)
  - `className/` - Class directories (mirrors MATLAB structure)
    - `__init__.py` - Class definition and method imports
    - `methodName.py` - Individual method implementations
- `Cora2025.1.0_Manual.txt` - CORA manual and documentation

## Translation Process Guide

### 1. Dependency Analysis
- `matlab_analysis_report.json`: Detailed dependency information
- `matlab_dependency_graph.gml`: Graph data for detailed analysis

### 2. Class Translation Order
Based on dependency analysis, translate classes from low dependencies to high

### 3. Translation Steps for Each Class

#### 3.1 Class Setup
1. Create class directory structure
2. Create `__init__.py` with class definition
3. Set up test directory structure
4. Create test `__init__.py` files

#### 3.2 Function-by-Function Translation
For each function in the class:

1. **Analysis**
   - Review MATLAB source in `cora_matlab/`
   - Check function documentation in manual
   - Study unit tests in `cora_matlab/unitTests/`
   - Review examples in `cora_matlab/examples/`

2. **Implementation**
   - Create function file (e.g., `functionName.py`)
   - Implement function with type hints and docstrings
   - Follow Python best practices
   - Add necessary imports

3. **Testing**
   - Port unit test to Python
   - Create test file (e.g., `test_functionName.py`)
   - Implement test cases from MATLAB
   - Add edge cases and error conditions

4. **Example Translation**
   - Port examples to Python
   - Create example file in `cora_python/examples/`
   - Verify example produces matching results

5. **Integration Testing**
   - Run unit tests
   - Run examples
   - Compare results with MATLAB
   - Document any differences

6. **Documentation**
   - Update Python docstrings
   - Add implementation notes
   - Document any MATLAB vs Python differences

### 4. Key Resources
- **Manual Sections**
  - Check manual for each class's mathematical foundation

- **Examples Location**
  - MATLAB: `cora_matlab/examples/`
  - Python: `cora_python/examples/`

- **Test Files**
  - MATLAB: `cora_matlab/unitTests/`
  - Python: `cora_python/tests/`

### 5. Translation Guidelines
1. **Maintain Consistency**
   - Keep method names similar where possible
   - Preserve mathematical operations
   - Match input/output behavior
   - Preserve comments about functionality
   - Preserve MATLAB-style directory structure

2. **Directory Structure**
   - Use `className/` directories for classes
   - Place class definition in `__init__.py`
   - Create separate files for each method
   - Import and attach methods in `__init__.py`
   - Example structure:
     ```
     cora_python/
     ├── contSet/
     │   ├── interval/
     │   │   ├── __init__.py
     │   │   ├── contains.py
     │   │   ├── is_intersecting.py
     │   │   └── ...
     │   └── ...
     └── ...
     ```

3. **Python Specifics**
   - Use NumPy for matrix operations
   - Implement operator overloading appropriately
   - Follow PEP 8 style guide
   - Add type hints (Python 3.7+)

4. **Performance Considerations**
   - Vectorize operations where possible
   - Use NumPy's optimized functions
   - Profile and optimize critical paths

### 6. Quality Assurance
1. Run unit tests after each function translation
2. Verify examples produce matching results
3. Document any precision differences

### 7. Function Translation Checklist (for each function)

For each function, perform the following steps and provide the required output:

1. Analyze the function file (paste summary here).
2. Analyze dependencies (list them here).
3. Read the corresponding manual entry (paste or summarize here, with section reference).
4. Look at examples (list example files and summarize their content).
5. Create the function file in Python (filename: ...).
6. Implement the function (paste code here).
7. Add type hints and docstrings (confirm done).
8. Create the test file (filename: ...).
9. Port unit tests (list test cases).
10. Port examples (list and paste code).
11. Run tests (paste results).
12. Run examples (paste results).
13. Compare with MATLAB (describe differences).
14. Document differences (paste here).
15. Update class __init__.py (confirm done).

## Notes
- The manual contains detailed mathematical descriptions
- Each class has corresponding documentation and examples
- Consider numerical precision differences between MATLAB and Python
- Maintain test coverage
- Document any implementation decisions or deviations
- Follow MATLAB-style directory structure
- Keep methods in separate files for better organization

## Your tasks
Translate the fuction plus from the interval class with its examples, tests, dependencies