# Upstream Computations Investigation Summary

## Status

I've added comprehensive tracking to compare upstream computations between Python and MATLAB:

1. **Tracking Added**:
   - `Z` before `quadMap`
   - `errorSec` after `quadMap` (before combining with `errorLagr`)
   - `errorLagr` before combining
   - `VerrorDyn` before and after reduction
   - `VerrorDyn` before `errorSolution_adaptive`
   - `Rerror` before `_aux_optimaldeltat`

2. **Files Modified**:
   - `cora_python/contDynamics/nonlinearSys/linReach_adaptive.py`: Added tracking for `VerrorDyn` and `Rerror`
   - `cora_python/contDynamics/nonlinearSys/private/priv_abstractionError_adaptive.py`: Added tracking for `Z`, `errorSec`, `errorLagr`, and `VerrorDyn`
   - `cora_matlab/contDynamics/@nonlinearSys/linReach_adaptive.m`: Added tracking for `VerrorDyn` and `Rerror`
   - `cora_matlab/contDynamics/@nonlinearSys/private/priv_abstractionError_adaptive.m`: Added tracking for `Z`, `errorSec`, `errorLagr`, and `VerrorDyn`

3. **Scripts Created**:
   - `track_upstream_python.py`: Runs Python with upstream tracking
   - `track_upstream_matlab.m`: Runs MATLAB with upstream tracking
   - `compare_upstream_computations.py`: Compares the tracked values

## Current Issue

The comparison script is having trouble matching MATLAB entries with Python entries. MATLAB generates 236 entries while Python generates 5401 entries, suggesting:
- MATLAB completes the simulation (237 steps)
- Python aborts early (many more inner loop iterations)

The MATLAB structured arrays need proper handling in the comparison script.

## Next Steps

1. Fix the comparison script to properly extract values from MATLAB structured arrays
2. Match entries by step number and run number
3. Compare the key values:
   - `errorSec` radius_max (from `quadMap`)
   - `VerrorDyn` radius_max (before and after reduction)
   - `Rerror` rerr1 (used in `_aux_optimaldeltat`)

## Key Findings So Far

- Python generates many more entries (5401 vs 236), indicating more inner loop iterations
- MATLAB completes the simulation while Python aborts early
- The tracking infrastructure is in place and working
