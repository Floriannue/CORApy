# Derive Function Dependency Check

## Status: ✅ `derive` TRANSLATED

The `derive` function has been **TRANSLATED** to Python.

## Location
- **MATLAB**: `cora_matlab/global/functions/verbose/write/derive.m`
- **Python**: ✅ **TRANSLATED** - `cora_python/g/functions/verbose/write/derive.py`

## Usage in hybridDynamics
- **Used in**: `cora_matlab/hybridDynamics/@nonlinearReset/derivatives.m`
- **Python equivalent**: ✅ **TRANSLATED** - `cora_python/hybridDynamics/nonlinearReset/derivatives.py`

## Dependencies of `derive` - ALL VERIFIED ✅

### ✅ Translated Dependencies
1. ✅ **writeMatrixFile** - EXISTS (`cora_python/g/functions/verbose/write/writeMatrixFile.py`)
2. ✅ **readNameValuePair** - EXISTS (`cora_python/g/functions/matlab/validate/preprocessing/readNameValuePair.py`)
3. ✅ **checkNameValuePairs** - EXISTS (`cora_python/g/functions/matlab/validate/check/checkNameValuePairs.py`)
4. ✅ **inputArgsCheck** - EXISTS (`cora_python/g/functions/matlab/validate/check/inputArgsCheck.py`)
5. ✅ **inputArgsLength** - EXISTS (`cora_python/g/functions/matlab/function_handle/inputArgsLength.py`)
6. ✅ **setDefaultValues** - EXISTS (`cora_python/g/functions/matlab/validate/preprocessing/setDefaultValues.py`)
7. ✅ **CORAROOT** - EXISTS (`cora_python/g/macros/CORAROOT.py`)
8. ✅ **CORAwarning** - EXISTS (`cora_python/g/functions/matlab/validate/postprocessing/CORAwarning.py`)
9. ✅ **CORAerror** - EXISTS (`cora_python/g/functions/matlab/validate/postprocessing/CORAerror.py`)

### ✅ Python Equivalents - ALL AVAILABLE
1. ✅ **sympy** - Python symbolic math library (equivalent to MATLAB Symbolic Toolbox)
   - `sympy.Symbol` - equivalent to `sym`
   - `sympy.Matrix.jacobian` - equivalent to `jacobian`
   - `sympy.subs` - equivalent to `subs`
2. ✅ **Function handle to string** - Python equivalent implemented using `inspect` and string manipulation
3. ✅ **Cell array operations** - Python list operations

## Auxiliary Functions in `derive.py` - ALL TRANSLATED ✅

### Main Function
- ✅ `derive(*varargin)` - Main function with name-value pairs

### Auxiliary Functions (all translated)
1. ✅ `aux_derive(f_sym, vars)` - Computes jacobian derivatives using sympy
2. ✅ `aux_getSymbolicFunction(f_sym, f, vars)` - Converts function handle to symbolic
3. ✅ `aux_setDefaultValues(vars, varNamesIn, varNamesOut, f)` - Sets default values

## Translation Status - COMPLETE ✅

1. ✅ **TRANSLATED**: `derive.m` → `derive.py`
   - Uses sympy instead of MATLAB Symbolic Toolbox
   - Implements `aux_derive`, `aux_getSymbolicFunction`, `aux_setDefaultValues`
   - All name-value pair parsing implemented
   - File generation using `writeMatrixFile`
   - Function handle evaluation implemented

2. ✅ **TRANSLATED**: `nonlinearReset.derivatives.m` → `nonlinearReset.derivatives.py`
   - Depends on `derive` (now translated)
   - Integrated into `__init__.py`
   - All tensor order support (1, 2, 3)

3. ✅ **VERIFIED**: All dependencies are available

## Summary

### ✅ All Dependencies Available
- All helper functions exist (readNameValuePair, checkNameValuePairs, inputArgsLength, writeMatrixFile)
- sympy is available for symbolic math (used in `contDynamics.derivatives` and `derive`)
- `derive` function is fully translated and functional
- `nonlinearReset.derivatives` method is fully translated and functional

### ✅ Complete
- **Status**: All `derive`-related functionality is now available
- **Core hybridDynamics**: All functionality is available
- **NonlinearReset**: Can now compute derivatives using `derive`
- **Integration**: Both functions are properly integrated into their respective `__init__.py` files

## Implementation Details

### Key Differences MATLAB → Python (All Handled)
1. **Symbolic Math**: MATLAB `sym` → Python `sympy.Symbol` ✅
2. **Jacobian**: MATLAB `jacobian(f, vars)` → Python `sympy.Matrix.jacobian` ✅
3. **Substitution**: MATLAB `subs(expr, old, new)` → Python `expr.subs(old, new)` ✅
4. **Function handles**: MATLAB `@(x) ...` → Python `lambda x: ...` ✅
5. **File generation**: MATLAB `.m` files → Python `.py` files ✅
6. **Name-value pairs**: MATLAB varargin → Python *varargin with readNameValuePair ✅
