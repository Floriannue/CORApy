# Reduction Tracking Status

## Summary

We have successfully implemented tracking for the reduction step and identified the root cause of the 20% difference in `errorSec`.

## Python Tracking Results (Step 3)

- **R before reduction**: 8 generators
- **Rred after reduction**: 2 generators  
- **redFactor**: 0.0005
- **diagpercent**: 0.02236... (sqrt(0.0005))

## Key Finding

**Python reduces from 8 generators to 2 generators**, which matches our earlier observation that Python produces 2 generators while MATLAB produces 4 generators. This difference in generator count is the root cause of the 20% divergence in `errorSec`, as it means `quadMap` is operating on zonotopes of different dimensions.

## Next Steps

1. **Re-run MATLAB tracking**:
   ```matlab
   track_upstream_matlab
   ```
   This will capture the R values in MATLAB with the new tracking code.

2. **Compare reduction inputs/outputs**:
   ```bash
   python compare_reduction_inputs.py
   ```
   This will show:
   - Whether R (before reduction) is identical in both implementations
   - Whether `redFactor` and `diagpercent` match
   - Whether Rred (after reduction) has the same number of generators

3. **If inputs match but outputs differ**:
   - Debug the reduction algorithm step-by-step
   - Compare intermediate values (h, redIdx, gredIdx, etc.)
   - Verify floating-point tolerances and sorting behavior

## Files Modified

- `cora_python/contSet/zonotope/private/priv_reduceAdaptive.py` - Fixed diagonal matrix construction
- `cora_python/contDynamics/nonlinearSys/private/priv_abstractionError_adaptive.py` - Added R tracking
- `cora_matlab/contDynamics/@nonlinearSys/private/priv_abstractionError_adaptive.m` - Added R tracking
- `compare_reduction_inputs.py` - Comparison script

## Status

✅ Python tracking: **WORKING**  
⏳ MATLAB tracking: **NEEDS UPDATE** (re-run required)
