# Step 2 Rlintp Divergence Analysis

## Summary

The root cause of Step 2's `Rtp` generator count difference has been identified:

### Key Finding

**Step 2 Run 2:**
- Python: `Rlintp` = 2 generators, `Rerror` = 10 generators → Expected `Rtp` = 12 generators
- MATLAB: `Rlintp` = 4 generators, `Rerror` = 10 generators → Expected `Rtp` = 14 generators
- **Difference**: `Rlintp` has 2 fewer generators in Python (2 vs 4)

### Root Cause

The divergence originates in **`initReach_adaptive`**:
- `Rlintp` comes from `initReach_adaptive`'s output `Rend.tp`
- Python's `Rend.tp` has 2 generators
- MATLAB's `Rend.tp` has 4 generators
- This 2-generator difference propagates to `Rtp = Rlintp + nlnsys.linError.p.x + Rerror`

### Important Note: Step 2's Code Path

For Step 2 Run 2, the `timeStepequalHorizon` path is used:
- `Rtp = Rtp_h + linx_h` (not `Rlintp + nlnsys.linError.p.x + Rerror`)
- `Rtp_h` comes from Step 1's Run 1 computation
- `Rlintp` in Step 2 Run 2 actually comes from Step 1's `initReach_adaptive` output (`Rend.tp`)

So the divergence in Step 2's `Rlintp` actually originates from **Step 1's `initReach_adaptive`**.

## Tracking Status

✅ **Python**: `initReach_tracking` is saved for Step 2 Run 2  
❌ **MATLAB**: `initReach_tracking` is not being saved correctly for Step 2 Run 2 (empty array in log)

The MATLAB tracking code is trying to save `initReach_tracking` but it's not working. The debug output shows that for other steps (233, 234), `initReach_tracking` is being saved successfully, but Step 2 is not appearing in the debug output.

## Next Steps

1. **Compare Step 1's `initReach_adaptive` outputs**:
   - `Rend.tp` (becomes `Rlintp`, which becomes `Rtp_h` for Step 2)
   - `Rend.ti` (becomes `Rlinti`, which becomes `Rti_h` for Step 2)
   - `Rstart` (input to `initReach_adaptive`)
   - `Rhom_tp` and `Rhom` (before reduction)

2. **Fix MATLAB tracking for Step 2**:
   - Investigate why `initReach_tracking` is not being saved for Step 2
   - The log entry update code might not be finding the correct entry
   - Or `initReach_adaptive` might not be called for Step 2 Run 2

3. **Trace the divergence chain**:
   - Step 1's `Rend.tp` → Step 2's `Rlintp` (via `Rtp_h`) → Step 2's `Rtp`
   - Compare each component to find where the 2-generator difference is introduced

## Current Data

### Python Step 2 Run 2:
- `Rlintp_tracking.num_generators`: 2
- `Rerror_tracking.num_generators`: 10
- `initReach_tracking`: Available (but not yet compared)

### MATLAB Step 2 Run 2:
- `Rlintp_tracking.num_generators`: 4
- `Rerror_tracking.num_generators`: 10
- `initReach_tracking`: Not available (empty array in log)
