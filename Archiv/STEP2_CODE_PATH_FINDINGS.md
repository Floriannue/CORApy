# Step 2 Code Path Analysis - Findings

## Key Discovery

**Step 2 uses the `timeStepequalHorizon` path!**

This means:
- Step 2's `Rtp = Rtp_h + linx_h + Rerror_h`
- `Rtp_h` comes from **Step 1's `Rlintp`** (stored during Step 1, Run 1)
- `Rerror_h` comes from **Step 1's `Rerror`** (stored during Step 1, Run 1)

## Current Status

1. ✅ **Fixed MATLAB tracking error**: Moved `Rend.ti/tp` tracking to after both are created
2. ✅ **Added code path tracking**: `timeStepequalHorizon_used` flag is now tracked
3. ✅ **Identified Step 2's path**: Step 2 uses `timeStepequalHorizon` path (Run 2)
4. ⚠️ **Tracking data incomplete**: Step 2's tracking fields exist but are empty arrays

## The Problem

Step 2's `Rtp` difference (Python 14 vs MATLAB 16 generators) comes from:
- **Step 1's `Rlintp`** (which becomes `Rtp_h`)
- **Step 1's `Rerror`** (which becomes `Rerror_h`)

Since Step 2 uses stored values from Step 1, the divergence must originate in **Step 1's computation**.

## Root Cause Chain

1. **Step 1**: `Rlintp` is computed (from `Rend.tp` in `initReach_adaptive`)
2. **Step 1**: `Rlintp` is stored as `Rtp_h` for Step 2
3. **Step 2**: Uses `Rtp_h` (from Step 1) → `Rtp = Rtp_h + linx_h + Rerror_h`
4. **Step 2**: `Rtp` has 14 vs 16 generators difference

## Next Steps

1. **Compare Step 1's `Rend.tp`**: This becomes `Rlintp`, which becomes `Rtp_h` for Step 2
2. **Compare Step 1's `Rerror`**: This becomes `Rerror_h` for Step 2
3. **Fix Step 1's computation**: Once we identify where Step 1 diverges, fix it

## Files Modified

- `cora_matlab/contDynamics/@linearSys/initReach_adaptive.m` - Fixed `Rend.ti/tp` tracking location
- `cora_python/contDynamics/nonlinearSys/linReach_adaptive.py` - Added `timeStepequalHorizon` path tracking
- `cora_matlab/contDynamics/@nonlinearSys/linReach_adaptive.m` - Added `timeStepequalHorizon` path tracking

## Scripts Created

- `check_step2_code_path.py` - Checks which code path Step 2 uses
- `check_all_steps_code_path.py` - Checks code paths for all steps
- `check_step2_fields.py` - Checks what fields Step 2 entries have
- `compare_step2_matlab_entry.py` - Analyzes Step 2's MATLAB entry
- `check_step1_for_step2.py` - Checks Step 1's values that become Step 2's inputs
