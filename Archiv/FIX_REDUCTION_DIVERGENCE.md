# Fix for Rlintp Generator Divergence

## Problem Summary

**Rlintp generator count divergence:**
- Python Step 2 Run 2: 2 generators
- MATLAB Step 2 Run 2: 4 generators

**Root cause chain:**
1. Python Step 2 Run 2 does NOT use `timeStepequalHorizon` path
2. Python calls `initReach_adaptive` with different `Rstart` than MATLAB
3. This leads to different `Rhom_tp` values
4. Different `Rhom_tp` → different `dHmax` (45x difference)
5. Different `dHmax` → different `redIdx` (3 vs 1)
6. Different `redIdx` → different final generator count (2 vs 4)

## Key Findings

1. **dHmax computation is correct** - Both Python and MATLAB use:
   ```
   dHmax = (diagpercent * 2) * sqrt(sum(Gbox.^2))
   ```

2. **Reduction algorithm logic is correct** - The indexing and array operations match MATLAB

3. **The divergence is in the INPUT to reduction**, not the reduction algorithm itself

## Solution

The fix needs to ensure Python Step 2 Run 2 uses the same path as MATLAB:
- MATLAB Step 2 Run 2 uses `timeStepequalHorizon` path (reuses Step 1 Run 1 results)
- Python Step 2 Run 2 should also use this path

**Next steps:**
1. Check why Python Step 2 Run 2 doesn't use `timeStepequalHorizon` path
2. Fix the condition that determines when to use this path
3. Ensure Python and MATLAB use the same `Rstart` values

## Current Status

- ✅ Reduction algorithm verified
- ✅ dHmax computation verified  
- ⏳ Need to fix `timeStepequalHorizon` path usage in Python
- ⏳ Need to ensure `Rstart` values match between Python and MATLAB
