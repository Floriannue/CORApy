# Fix Divergence Summary

## Root Cause Identified

The divergence starts at **Step 2's Rtp computation** and propagates through all subsequent steps.

### Divergence Chain:

1. **Step 1's Rtp AFTER reduction** → becomes **Step 2's Rstart**
   - Need to compare: Python vs MATLAB

2. **Step 2's Rtp BEFORE reduction**: Python 14, MATLAB 16 (difference: 2 generators)
   - This is computed as: `Rtp = Rlintp + nlnsys.linError.p.x + Rerror`
   - Components to check:
     - `Rlintp` (from `initReach_adaptive` as `Rend.tp`)
     - `Rerror` (from `errorSolution_adaptive`)

3. **Step 2's Rtp AFTER reduction**: Python 2, MATLAB 4 (difference: 2 generators)
   - This becomes **Step 3's Rstart**

4. **Step 3's Rlintp**: Python 2, MATLAB 4 (difference: 2 generators)
   - This comes from `Rend.tp` in `initReach_adaptive`
   - **Rhom_tp** (input to Rend.tp reduction): Python 5, MATLAB 7 (difference: 2 generators)
   - **Rhom** (before reduction): Python 15, MATLAB 21 (difference: 6 generators)

## Key Finding

The 2-generator difference appears consistently:
- Step 2's Rtp: 14 vs 16 (difference: 2)
- Step 2's Rtp after reduction: 2 vs 4 (difference: 2)
- Step 3's Rstart: 2 vs 4 (difference: 2)
- Step 3's Rhom_tp: 5 vs 7 (difference: 2)
- Step 3's Rlintp: 2 vs 4 (difference: 2)

## Next Steps

1. **Compare Step 1's Rtp after reduction** to see if that's where it starts
2. **Compare Step 2's Rerror** to see if that contributes to the difference
3. **Fix the Python code** to match MATLAB once the exact source is identified

## Files Modified

- Added tracking for `Rlintp_tracking` and `Rerror_tracking` in `linReach_adaptive`
- Added tracking for `Rtp_before_reduction` and `Rtp_after_reduction` in `reach_adaptive`
- Comparison scripts created to identify the divergence point
