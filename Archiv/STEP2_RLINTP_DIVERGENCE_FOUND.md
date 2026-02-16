# Step 2 Rlintp Divergence Found

## Summary

The root cause of Step 2's `Rtp` generator count difference has been identified.

## Comparison Results

### Step 2 Run 2 Components:

| Component | Python | MATLAB | Status |
|-----------|--------|--------|--------|
| `Rlintp` | 2 generators | 4 generators | **MISMATCH** (diff: 2) |
| `Rerror` | 10 generators | 10 generators | **MATCH** |
| Expected `Rtp` | 12 generators | 14 generators | **MISMATCH** (diff: 2) |

### Actual Rtp (before reduction in reach_adaptive):

- Python: 14 generators (observed earlier)
- MATLAB: 16 generators (observed earlier)
- Difference: 2 generators

**Note**: The actual `Rtp` has 2 more generators than expected from `Rlintp + Rerror` in both cases. This suggests the linearization point translation (`nlnsys.linError.p.x`) or some other operation is adding 2 generators.

## Root Cause

The divergence originates in **`initReach_adaptive`**:
- `Rlintp` comes from `initReach_adaptive`'s output `Rend.tp`
- Python's `Rend.tp` has 2 generators
- MATLAB's `Rend.tp` has 4 generators
- This 2-generator difference propagates to `Rtp = Rlintp + nlnsys.linError.p.x + Rerror`

## Next Steps

1. Compare `initReach_adaptive`'s inputs for Step 2:
   - `Rstart` (input to `initReach_adaptive`)
   - `Rdelta = Rstart + (-nlnsys.linError.p.x)`
   - System matrices and parameters

2. Compare `initReach_adaptive`'s intermediate values:
   - `Rhom_tp` (before reduction)
   - `Rhom` (before reduction)
   - Reduction parameters (`redFactor`, etc.)

3. Compare `initReach_adaptive`'s outputs:
   - `Rend.tp` (becomes `Rlintp`)
   - `Rend.ti` (becomes `Rlinti`)

4. Trace back to Step 1's `Rtp` (which becomes Step 2's `Rstart`) to see if the divergence starts there.

## Tracking Status

✅ **Fixed**: MATLAB tracking for `Rlintp_tracking` and `Rerror_tracking`  
✅ **Fixed**: Python tracking for `Rlintp_tracking` and `Rerror_tracking`  
✅ **Working**: Both Python and MATLAB now capture these values correctly
