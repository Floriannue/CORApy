# Rlinti 2 vs 7 Generators Difference - Summary

## The Finding

**Step 3's Rlinti** (from `Rend.ti` in `initReach_adaptive`):
- **Python**: 2 generators
- **MATLAB**: 7 generators
- **Difference**: 5 generators

This is the **"init reach 2 vs 7 generators difference"** you asked about.

## Where It Comes From

1. **Rend.ti** is computed in `initReach_adaptive` by reducing `Rhom` (the homogeneous solution)
2. **Rend.ti** becomes **Rlinti** in `linReach_adaptive`
3. **Rlinti** is used in: `Rmax = Rlinti + RallError`
4. The difference propagates through the entire computation chain

## Divergence Chain

From `DIVERGENCE_ROOT_CAUSE_ANALYSIS.md`:

1. **Step 3's Rstart**: Python 2, MATLAB 4 ❌ (root cause)
2. **Rdelta** (input to `initReach_adaptive`): Python 2, MATLAB 4 ❌
3. **Rhom** (before reduction): Python 15, MATLAB 21 ❌
4. **Rend.ti** (after reduction, becomes Rlinti): **Python 2, MATLAB 7** ❌ ← **THIS IS THE 2 vs 7 DIFFERENCE**
5. **Rmax = Rlinti + RallError**: Python 8, MATLAB 13 ❌
6. **Rred** (after reduction): Python 2, MATLAB 4 ❌

## Why This Happens

The reduction in `initReach_adaptive` for `Rend.ti` produces different results:
- Python reduces `Rhom` (15 generators) → `Rend.ti` (2 generators)
- MATLAB reduces `Rhom` (21 generators) → `Rend.ti` (7 generators)

The input `Rhom` already differs (15 vs 21), which is why the output differs.

## Root Cause

The root cause is **Step 3's Rstart** (2 vs 4 generators), which comes from:
- Step 2's `Rtp` after reduction in `reach_adaptive`

This propagates through:
- `Rstart` → `Rdelta` → `Rhom` → `Rend.ti` → `Rlinti` → `Rmax`

## Current Status

✅ **Identified**: The 2 vs 7 difference is in `Rend.ti` / `Rlinti`
✅ **Traced**: The divergence chain is fully mapped
⏳ **Next Step**: Fix the root cause at Step 2's `Rtp` computation

## Files to Check

1. `cora_python/contDynamics/linearSys/initReach_adaptive.py` - Reduction of `Rhom` to `Rend.ti`
2. `cora_matlab/contDynamics/@linearSys/initReach_adaptive.m` - MATLAB's reduction
3. Compare the reduction inputs (`Rhom`) and `redFactor` to ensure they match
