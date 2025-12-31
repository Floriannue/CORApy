# Derivatives Translation Plan

This document tracks the translation of `contDynamics.derivatives` and all its dependencies.

## Main Function
- [ ] `contDynamics.derivatives` - Main code generation function

## Dependencies (in order of translation)

### Phase 1: Helper Functions
- [ ] `getDefaultValue` - Wrapper for getDefaultValueOptions/Params
- [ ] `getDefaultValueOptions` - Default values for options struct
- [ ] `getDefaultValueParams` - Default values for params struct
- [ ] `unitvector` - Creates unit vector
- [ ] `bracketSubs` - Substitutes brackets in symbolic expressions
- [ ] `isequalFunctionHandle` - Compares function handles
- [ ] `rmiffield` - Removes field from struct if it exists
- [ ] `CORAVERSION` - Returns CORA version string
- [ ] `canUseParallelPool` - Checks if parallel pool can be used (may not exist)

### Phase 2: Symbolic Variable Creation
- [ ] `symVariables` - Creates symbolic variables for system

### Phase 3: Tensor Recomputation Check
- [ ] `priv_checkTensorRecomputation` - Checks if recomputation needed

### Phase 4: File Writing Functions
- [ ] `writeMatrix` - Writes a single matrix to file
- [ ] `writeMatrixFile` - Writes multiple matrices to .m/.py file
- [ ] `writeHessianTensorFile` - Writes Hessian tensor to file
- [ ] `write3rdOrderTensorFile` - Writes third-order tensor to file
- [ ] `writeHigherOrderTensorFiles` - Writes higher-order tensors to file

### Phase 5: Main Function
- [ ] `derivatives` - Main function with all auxiliary functions

## Notes
- All functions use sympy for symbolic math (instead of MATLAB Symbolic Toolbox)
- Generated files are Python .py files (instead of MATLAB .m files)
- Function handles in Python use string-based evaluation or importlib
- File paths use os.path.join instead of filesep

## Status
Starting translation...

