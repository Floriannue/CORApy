# MATLAB Test Status

## Current Status

✅ **MATLAB tracking code implemented** - All tracking code added to MATLAB files
⚠️ **MATLAB test needs default options** - Test script requires CORA's default options system

## What's Ready

1. **Tracking Code**: 
   - ✅ `linReach_adaptive.m` - Tracking code added
   - ✅ `priv_abstractionError_adaptive.m` - Z/errorSec tracking added
   - ✅ All calls updated to pass `trace_file` parameter

2. **Test Scripts**:
   - ✅ `test_tracking_jetEngine_matlab.m` - Basic test (needs options fix)
   - ✅ `test_tracking_jetEngine_matlab_complete.m` - Complete test with error handling

## Issue

The MATLAB test fails because CORA requires default options to be set. The test script tries to set minimal options, but CORA may need additional fields.

## Solutions

### Option 1: Use Existing CORA Example

1. Find a working CORA example that uses `reach_adaptive` with `jetEngine`
2. Add `options.traceIntermediateValues = true;` 
3. Set `params.tFinal = 5.0;`
4. Run the modified example

### Option 2: Set All Required Options

The test script needs these options (in addition to what's already set):
- `options.isHessianConst = false;`
- `options.hessianCheck = false;`
- Plus any other defaults that CORA's `reach_adaptive` expects

### Option 3: Use CORA's Option System

If CORA has a function to get default options (like `CORAoptions()`), use it:
```matlab
options = CORAoptions();
options.alg = 'lin-adaptive';
options.traceIntermediateValues = true;
options.tFinal = 5.0;
```

## Once MATLAB Test Runs Successfully

1. **Compare growth patterns**: Check if MATLAB shows same 8.27 trillion times growth
2. **Compare Step 451**: Use `compare_intermediate_values.py` to find divergence
3. **Analyze error_adm_horizon source**: Verify if it's set from run == 2's trueError

## Python Results (for comparison)

- **897 steps** tracked
- **Growth**: 5.34e-02 → 4.42e+11 (8.27 trillion times)
- **First explosion**: Step 451
- **Key finding**: Step 450 → 451 shows 3.4x jump (not from trueError)

## Next Steps

1. ⏳ **Fix MATLAB test options** - Get default options working
2. ⏳ **Run MATLAB test** - Generate MATLAB trace files
3. ⏳ **Compare Step 451** - Find where values diverge
4. ⏳ **Analyze error_adm_horizon update** - Verify source of the jump
