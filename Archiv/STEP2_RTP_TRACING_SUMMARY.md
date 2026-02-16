# Step 2 Rtp Tracing Summary

## Current Status

**Step 2's Rtp BEFORE reduction**: Python 14, MATLAB 16 (difference: 2 generators)
**Step 2's Rtp AFTER reduction**: Python 2, MATLAB 4 (difference: 2 generators)

## Problem

Step 2's tracking data is **not available** in the logs:
- ❌ No `initReach_tracking` for Step 2
- ❌ No `Rlintp_tracking` for Step 2
- ❌ No `Rerror_tracking` for Step 2
- ❌ No `Rtp_final_tracking` for Step 2

This suggests Step 2 might use a **different code path** than Step 3+.

## Possible Code Paths

From `linReach_adaptive.py`, there are two paths for computing `Rtp`:

### Path 1: `timeStepequalHorizon == True`
```python
Rtp = Rtp_h + linx_h
Rerror = Rerror_h
```
This uses stored values from a previous run (`Rtp_h`, `Rerror_h`).

### Path 2: `timeStepequalHorizon == False` (normal path)
```python
Rtp = Rlintp + nlnsys.linError.p.x
Rtp = Rtp + Rerror
```
This computes `Rtp` from `Rlintp` (from `initReach_adaptive`) and adds `Rerror`.

## Hypothesis

**Step 2 might be using Path 1** (`timeStepequalHorizon == True`), which means:
- `Rtp = Rtp_h + linx_h`
- `Rtp_h` comes from Step 1's `Rlintp`
- The difference might be in Step 1's `Rlintp` or in how `Rtp_h` is stored

## Next Steps

1. **Check if Step 2 uses `timeStepequalHorizon` path**: Add debug output to see which path Step 2 takes
2. **Compare Step 1's Rlintp**: If Step 2 uses `Rtp_h`, we need to check Step 1's `Rlintp`
3. **Add tracking for Step 2**: Ensure tracking is enabled for Step 2's code path
4. **Compare Step 1's Rtp**: Check if Step 1's `Rtp` (which becomes `Rtp_h`) differs between Python and MATLAB

## Files to Check

1. `cora_python/contDynamics/nonlinearSys/linReach_adaptive.py` - Check `timeStepequalHorizon` logic
2. `cora_matlab/contDynamics/@nonlinearSys/linReach_adaptive.m` - Check MATLAB's equivalent logic
3. Add debug output to identify which path Step 2 uses

## Comparison with Step 3

Step 3 has full tracking:
- ✅ `Rlintp_tracking`: Python 2, MATLAB 4 (difference: 2)
- ✅ `Rerror_tracking`: Not saved (bug in tracking code)
- ✅ `Rtp_final_tracking`: Available
- ✅ `Rtp before reduction`: Python 14, MATLAB 16 (difference: 2)

The pattern suggests:
- `Rlintp` differs by 2 generators (2 vs 4)
- `Rerror` might also differ
- The sum `Rlintp + Rerror` = `Rtp` differs by 2 generators (14 vs 16)

## Action Items

1. **Fix Rerror_tracking**: Ensure `Rerror_tracking` is saved to the log (currently it's set but not saved)
2. **Add Step 2 tracking**: Ensure tracking works for Step 2's code path
3. **Compare Step 1**: If Step 2 uses `Rtp_h`, compare Step 1's `Rlintp` and `Rtp`
4. **Debug code path**: Add logging to identify which path Step 2 uses
