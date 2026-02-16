# MATLAB Intermediate Value Tracking Implementation

## Status: ✅ COMPLETE

MATLAB equivalent tracking code has been implemented to match the Python implementation.

## What Was Implemented

### 1. MATLAB Tracking Code

**File: `cora_matlab/contDynamics/@nonlinearSys/linReach_adaptive.m`**
- Added `traceIntermediateValues` option support
- Opens trace file at start of inner loop
- Tracks all key values in inner loop:
  - `error_adm` (input and updated)
  - `RallError` (center, radius, radius_max)
  - `Rmax` (center, radius, radius_max)
  - `VerrorDyn` (center, radius, radius_max)
  - `trueError` (vector and max)
  - `perfIndCurr` (value, Inf/NaN status, comparison)
  - `perfInds` (array for divergence detection)
  - Convergence/divergence status
  - Algorithm and tensorOrder information
- Closes trace file at end of inner loop

**File: `cora_matlab/contDynamics/@nonlinearSys/private/priv_abstractionError_adaptive.m`**
- Added `trace_file` parameter (optional, 13th parameter)
- Added tracking for `Z` (before quadMap) - only for tensorOrder == 3
- Added tracking for `errorSec` (after quadMap) - only for tensorOrder == 3
- All errors are logged (no silent failures)
- Uses `nargin >= 13` to check if trace_file is provided

### 2. Test Script

**File: `test_tracking_jetEngine_matlab.m`**
- MATLAB test script that enables tracking and runs jetEngine case
- Verifies trace files are created
- Shows summary of tracked values

## Key Differences from Python

1. **File I/O**: MATLAB uses `fopen`/`fprintf`/`fclose` instead of Python's `open`/`write`/`close`
2. **String formatting**: MATLAB uses `mat2str()` and `sprintf()` instead of f-strings
3. **Parameter checking**: MATLAB uses `nargin >= 13` to check if optional `trace_file` is provided
4. **Flush**: MATLAB uses `fflush(trace_file)` instead of Python's `flush()`

## Usage

**Enable tracking:**
```matlab
options.traceIntermediateValues = true;
```

**Run test:**
```matlab
test_tracking_jetEngine_matlab
```

**Compare traces:**
```bash
python compare_intermediate_values.py matlab_trace.txt python_trace.txt 1e-10
```

## Files Modified

1. `cora_matlab/contDynamics/@nonlinearSys/linReach_adaptive.m`
   - Added trace file opening/closing
   - Added tracking for all intermediate values
   - Updated calls to `priv_abstractionError_adaptive` to pass `trace_file`

2. `cora_matlab/contDynamics/@nonlinearSys/private/priv_abstractionError_adaptive.m`
   - Added `trace_file` parameter
   - Added Z and errorSec tracking
   - Updated function signature and documentation

3. `test_tracking_jetEngine_matlab.m` (new)
   - Test script for MATLAB tracking

## Notes

- **TensorOrder dependency**: Z and errorSec are only tracked for tensorOrder == 3
- **File naming**: Files are named `intermediate_values_step{N}_inner_loop.txt` (same as Python)
- **Error handling**: All tracking errors are logged (no silent failures)
- **Backward compatibility**: All existing calls to `priv_abstractionError_adaptive` updated to pass empty `[]` for `trace_file` when not tracking

## Verification

The MATLAB implementation mirrors the Python implementation exactly:
- Same values tracked
- Same file format
- Same error handling approach
- Ready for side-by-side comparison
