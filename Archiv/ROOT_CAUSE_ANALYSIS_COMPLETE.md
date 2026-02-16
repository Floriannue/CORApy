# Root Cause Analysis: Complete Findings

## Executive Summary

The generator count divergence (Python: 2 vs MATLAB: 4) is **NOT** a reduction algorithm bug. It is caused by a **cascade of differences** starting from different `Rstart` values between Run 1 and Run 2, leading to different time step selection and ultimately different reduction results.

## The Cascade

```
Different Rstart values (Run 1 vs Run 2)
  ↓
Different timeStep selected by optimaldeltat
  ↓
Different eAt = expm(A * timeStep)
  ↓
Different Rtrans and inputCorr
  ↓
Different Rhom_tp generator values (same count, different magnitudes)
  ↓
Different reduction results (2 vs 4 generators)
```

## Key Statistics

### Run 1 (Matches MATLAB)
- **Mean timeStep**: 0.005633
- **Mean Rstart_norm**: 0.006312
- **Rend_tp generators**: {2, 3, 4} (mostly 4)
- **Pattern**: Larger Rstart → Larger timeStep → Conservative reduction (4 generators)

### Run 2 (Differs from MATLAB)
- **Mean timeStep**: 0.001856 (67% smaller than Run 1)
- **Mean Rstart_norm**: 0.002281 (64% smaller than Run 1)
- **Rend_tp generators**: {2} (always 2)
- **Pattern**: Smaller Rstart → Smaller timeStep → Aggressive reduction (2 generators)

### Direct Comparison (Step 4 Example)
- **Run 1**: Rstart_norm=0.030365, timeStep=0.016530, Rend_tp=4 gens
- **Run 2**: Rstart_norm=0.017930, timeStep=0.009761, Rend_tp=2 gens
- **Difference**: Rstart_norm 41% smaller, timeStep 41% smaller

## The Mystery: Why Does Run 2 Have Different Rstart?

**Expected Behavior**: Run 2 should use Run 1's results via `timeStepequalHorizon` path, meaning `Rstart` should be the same or very similar.

**Actual Behavior**: Run 2 has significantly smaller `Rstart` values, suggesting:
1. The `timeStepequalHorizon` path isn't working correctly in Python
2. `Rstart` is being computed/updated differently in Run 2
3. There's a bug in how `Rstart` is passed between runs

## Investigation Points

### 1. Check timeStepequalHorizon Path
- Verify that Python Run 2 actually uses Run 1's results
- Compare `Rlintp` values between Run 1 and Run 2
- Check if `Rstart` is correctly set from `Rlintp` in Run 2

### 2. Compare Rstart Computation
- Check how `Rstart` is computed/updated in `linReach_adaptive`
- Verify if `Rstart` is modified between Run 1 and Run 2
- Compare `Rstart` values between Python Run 2 and MATLAB Run 2

### 3. Verify optimaldeltat Implementation
- Ensure `_aux_optimaldeltat` produces same output for same inputs
- Compare `Rstart`, `Rerror_h`, `finitehorizon`, `varphi`, `zetaP` between Python and MATLAB
- Check if floating-point precision differences affect time step selection

## Files to Investigate

1. **`cora_python/contDynamics/nonlinearSys/linReach_adaptive.py`**
   - `timeStepequalHorizon` path (around line 700)
   - How `Rstart` is set for Run 2
   - `_aux_optimaldeltat` call (line 578)

2. **`cora_matlab/contDynamics/@nonlinearSys/linReach_adaptive.m`**
   - `timeStepequalHorizon` path
   - How `Rstart` is set for Run 2
   - `aux_optimaldeltat` call

3. **`cora_python/contDynamics/nonlinearSys/linReach_adaptive.py`**
   - `_aux_optimaldeltat` implementation (line 946)

## Status

✅ **ROOT CAUSE IDENTIFIED**: Time step divergence due to different Rstart values  
⏳ **NEXT STEP**: Investigate why Run 2 has different Rstart values  
⏳ **HYPOTHESIS**: `timeStepequalHorizon` path may not be working correctly in Python

## Conclusion

The reduction algorithm is working correctly. The divergence occurs because:
1. Run 2 receives different `Rstart` values than Run 1
2. This leads to different time step selection
3. Different time steps produce different `Rhom_tp` generator values
4. Different generator values lead to different reduction results

The fix should focus on ensuring `Rstart` is correctly set in Run 2, not on modifying the reduction algorithm.
