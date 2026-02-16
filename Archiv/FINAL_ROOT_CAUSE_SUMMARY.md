# Final Root Cause Summary

## Confirmed Root Cause

**The generator count divergence (Python: 2 vs MATLAB: 4) is caused by different time step values selected in the adaptive loop, which stems from different `Rstart` values between Run 1 and Run 2.**

## The Complete Cascade

```
Different Rstart values (Run 1 vs Run 2)
  ↓
Different timeStep selected by _aux_optimaldeltat()
  ↓
Different eAt = expm(A * timeStep)
  ↓
Different Rtrans and inputCorr (depend on timeStep)
  ↓
Different Rhom_tp generator values (same count=5, different magnitudes)
  ↓
Different reduction results (2 vs 4 generators)
```

## Key Evidence

### Statistical Analysis
- **Run 1** (matches MATLAB): Mean timeStep=0.0056, Mean Rstart_norm=0.0063 → 4 generators
- **Run 2** (differs): Mean timeStep=0.0019, Mean Rstart_norm=0.0023 → 2 generators
- **For same step**: Run 2 has 40-50% smaller Rstart_norm → 40-50% smaller timeStep

### Direct Comparison (Step 4)
- **Run 1**: Rstart_norm=0.030365, timeStep=0.016530, Rend_tp=4 gens
- **Run 2**: Rstart_norm=0.017930, timeStep=0.009761, Rend_tp=2 gens
- **Difference**: 41% smaller Rstart_norm → 41% smaller timeStep

## Why Rstart Differs

**Expected**: Run 2 should use Run 1's results via `timeStepequalHorizon` path, so `Rstart` should be similar.

**Actual**: Run 2 has significantly smaller `Rstart` values.

**Possible Causes**:
1. `Rstart` for Run 2 is set to reduced `Rtp` from Run 1 (`options['R'] = Rnext['tp']` at line 156 of `reach_adaptive.py`)
2. The reduction in Run 1 changes the center of `Rtp`, making `Rstart` for Run 2 different
3. The `timeStepequalHorizon` path may not be working correctly, causing Run 2 to use different `Rstart`

## Code Flow

1. **Step 1 Run 1**: 
   - `Rstart` = initial `R0`
   - Calls `initReach_adaptive` → produces `Rtp`
   - Reduces `Rtp` → `options['R'] = reduced Rtp`

2. **Step 1 Run 2**:
   - `Rstart` = reduced `Rtp` from Run 1 (should have 4 generators, but tracking shows 5?)
   - If `timeStepequalHorizon`: uses `Rtp_h` from Run 1, doesn't call `initReach_adaptive`
   - If NOT `timeStepequalHorizon`: calls `initReach_adaptive` with different `Rstart`

3. **Step 2 Run 2**:
   - `Rstart` = reduced `Rtp` from Step 1 Run 2
   - This `Rstart` has different center than Step 1 Run 1's `Rstart`
   - Leads to different time step selection

## Next Investigation Steps

1. **Verify `timeStepequalHorizon` logic**: Check if Run 2 actually uses this path or calls `initReach_adaptive`
2. **Compare `Rstart` values**: Check if `Rstart` for Run 2 matches what it should be (reduced `Rtp` from Run 1)
3. **Check reduction impact**: Verify if reduction changes the center of `Rtp`, affecting `Rstart` for next run
4. **Compare with MATLAB**: Check if MATLAB Run 2 has the same `Rstart` values as Python Run 2

## Files Modified

✅ `cora_matlab/contDynamics/@linearSys/initReach_adaptive.m` - Fixed structure concatenation error
✅ `compare_reduction_params.py` - Fixed comparison logic

## Status

🔴 **ROOT CAUSE IDENTIFIED**: Time step divergence due to different Rstart values  
⏳ **INVESTIGATION ONGOING**: Why does Run 2 have different Rstart values?  
✅ **REDUCTION ALGORITHM**: Confirmed working correctly - not the source of the bug

## Conclusion

The reduction algorithm (`priv_reduceAdaptive`) is working correctly. The divergence occurs because:
1. Run 2 receives different `Rstart` values than Run 1
2. This leads to different time step selection via `_aux_optimaldeltat`
3. Different time steps produce different `Rhom_tp` generator values
4. Different generator values lead to different reduction results (2 vs 4 generators)

**The fix should focus on ensuring `Rstart` is correctly set/used in Run 2, not on modifying the reduction algorithm.**
