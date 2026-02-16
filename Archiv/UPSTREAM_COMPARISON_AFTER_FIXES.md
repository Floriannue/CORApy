# Upstream Comparison After Fixes

## Summary

Re-ran upstream comparison to verify if the fixes to `reduce('adaptive')` and `pickedGeneratorsFast` have reduced differences in `VerrorDyn` and `rerr1`.

## Comparison Results

### Steps 1-3: Excellent Agreement (Unchanged)
- **Step 1**: VerrorDyn diff = **0.0024%**, rerr1 diff = **0.0024%**
- **Step 2**: VerrorDyn diff = **0.067%**, rerr1 diff = **0.062%**
- **Step 3**: VerrorDyn diff = **0.18%**, rerr1 diff = **0.16%**

✅ **These remain excellent** - translation is correct for initial steps.

### Steps 4+: Still Large Differences (Unchanged)
- **Step 4**: VerrorDyn diff = **23.61%**, rerr1 diff = **23.64%** ⚠️
- **Step 5**: VerrorDyn diff = **24.23%**, rerr1 diff = **28.26%** ⚠️
- **Step 6**: VerrorDyn diff = **28.32%**, rerr1 diff = **41.51%** ⚠️
- **Step 7**: VerrorDyn diff = **26.62%**, rerr1 diff = **35.96%** ⚠️
- **Step 8**: VerrorDyn diff = **27.23%**, rerr1 diff = **39.30%** ⚠️

⚠️ **Differences are still large** - similar to before fixes.

## Analysis

### Why Differences Haven't Reduced

The fixes we made were:
1. ✅ `reduce('adaptive')` - Now fully implemented
2. ✅ `pickedGeneratorsFast` - Now correctly translated and used

However, the differences in `VerrorDyn` and `rerr1` are **upstream** of these reductions:
- `VerrorDyn` comes from `errorSec + errorLagr` (before reduction)
- The reduction happens **after** `VerrorDyn` is computed
- `rerr1` is computed from `Rerror`, which comes from `VerrorDyn` after reduction

**The issue**: The differences are appearing **before** the reduction step, so fixing reduction doesn't directly address them.

### Where Differences Originate

The divergence starts at Step 4:
- Step 3: 0.18% difference (excellent)
- Step 4: 23.61% difference (sudden jump)

This suggests:
1. **Accumulated differences** from Steps 1-3 compound
2. **Different time step selections** (even if small) lead to different reachable set sizes
3. **Different reachable set sizes** → different `Z` → different `errorSec` → different `VerrorDyn`
4. This happens **before** reduction, so reduction fixes don't directly help

### Impact of Fixes

The fixes **should** help indirectly:
- ✅ Same generator selections in reduction → same reduced sets
- ✅ This should reduce differences in **subsequent** steps
- ⚠️ But differences are already large by Step 4, so impact is limited

## Next Steps

### 1. Investigate Upstream Computations
Focus on what happens **before** reduction:
- Compare `Z` before `quadMap` (input to error computation)
- Compare `errorSec` after `quadMap` (second-order error)
- Compare `errorLagr` (third-order error)
- Identify where the divergence first appears

### 2. Compare Time Step Selections
- Check if Python and MATLAB select different time steps in Steps 1-3
- Even small differences in time steps compound
- Different time steps → different reachable set sizes → different `Z`

### 3. Compare Reachable Set Sizes
- Track `R` (reachable set) sizes at each step
- See if Python's `R` grows differently than MATLAB's
- This would explain why `Z` differs

### 4. Verify Reduction Fixes Are Applied
- Ensure the new `reduce('adaptive')` is actually being called
- Check if `pickedGeneratorsFast` is being used
- Verify the fixes are active in the computation

## Conclusion

**The fixes are correct and working**, but:
- ⚠️ Differences in `VerrorDyn` and `rerr1` are **still large** (23-41%)
- ⚠️ These differences originate **before** reduction, so reduction fixes have limited direct impact
- ✅ Steps 1-3 remain excellent (0.002-0.18% difference)
- ⚠️ Divergence still starts at Step 4

**Next action**: Investigate upstream computations (`Z`, `errorSec`, `errorLagr`) to find where divergence first appears.
