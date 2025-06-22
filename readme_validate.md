# CORA Translation Project: AI Assistant Instructions

You are an advanced AI assistant acting as a professional software engineer. Your primary goal is to translate MATLAB code from the CORA library (`cora_matlab/`) to Python (`cora_python/`) file by file, following these instructions precisely. You provide high-quality, well-documented, and tested Python code that mirrors the structure and functionality of the original MATLAB code.

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
│   ├── g/        # mirrors cora_matlab/global
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

> You **must** read the readme.md for rules on how to do it but ignore the Task inside

## Validate and Fix Steps (for already translated parts)
- identify which already translated parts are missing tests
- translate all the missing tests from matlab to python. If test cases are missing implement them so that everything is fully tested!
- Ensure all methods are correctly attached in the __init__.py except for methods that have to be in the main class file - like abstract inherited methods 
- Remove lazy imports and use normal ones. If there is a circular import problem use the typing module. Only use lazy imports as last resort
- compare against the matlab implementation and the cora manual - ensure everything is fully and accurately translated
- run the test (`pytest`) - fix errors by looking at how it is done in matlab and expected by the manual





## Task
Your task is to validate and fix `ellipsoid`