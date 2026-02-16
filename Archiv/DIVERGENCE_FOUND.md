# Divergence Root Cause Found

## Summary
The divergence between Python and MATLAB starts at **Step 2's Rtp computation** in `linReach_adaptive`.

## Key Findings

### Step 2 - Rtp in reach_adaptive:
1. **Rtp BEFORE reduction**:
   - Python: **14 generators**
   - MATLAB: **16 generators**
   - **MISMATCH: 14 vs 16** ❌
   - redFactor: Both 0.0005 ✅

2. **Rtp AFTER reduction**:
   - Python: **2 generators**
   - MATLAB: **4 generators**
   - **MISMATCH: 2 vs 4** ❌
   - This becomes **Step 3's Rstart**!

### Step 3 - Rstart (input to linReach_adaptive):
- Python: **2 generators**
- MATLAB: **4 generators**
- **MISMATCH: 2 vs 4** ❌

## Root Cause

The divergence occurs in **Step 2's Rtp computation** in `linReach_adaptive`. The Rtp BEFORE reduction has different generator counts (14 vs 16), which leads to different reduction results (2 vs 4), which then becomes Step 3's Rstart.

## Next Steps

1. **Trace Step 2's Rtp computation**: Compare how `Rtp` is computed in `linReach_adaptive` for Step 2
   - `Rtp = Rlintp + nlnsys.linError.p.x + Rerror`
   - Check `Rlintp` (from `initReach_adaptive`)
   - Check `Rerror` (from `errorSolution_adaptive`)
   - Check `nlnsys.linError.p.x` (linearization point)

2. **Compare Step 1's output**: Verify if Step 1's Rtp after reduction matches between Python and MATLAB (this becomes Step 2's Rstart)

3. **Fix the divergence**: Once identified, fix the Python code to match MATLAB's computation

## Files Modified

- `cora_python/contDynamics/nonlinearSys/reach_adaptive.py` - Added Rtp tracking
- `cora_matlab/contDynamics/@nonlinearSys/reach_adaptive.m` - Added Rtp tracking
- `compare_Rtp_reach_adaptive.py` - Comparison script
