# MATLAB debug_step Fix

## Issue

MATLAB was throwing an error:
```
Unrecognized property 'debug_step' for class 'zonotope'.
Error in linearSys/initReach_adaptive (line 129)
    Rhom_tp.debug_step = options.i;
```

## Root Cause

MATLAB `zonotope` objects don't support dynamic property assignment. You cannot add arbitrary properties to MATLAB class instances like you can in Python.

## Fix

Removed all attempts to set `debug_step` and `debug_run` properties on zonotope objects in `initReach_adaptive.m`:

1. **Line 129-130**: Removed `Rhom_tp.debug_step` and `Rhom_tp.debug_run` assignments
2. **Line 148-149**: Removed `Rhom_tp.debug_step` and `Rhom_tp.debug_run` assignments
3. **Line 137-142**: Removed cleanup code for these fields (no longer needed)

## Impact

The reduction tracking in `priv_reduceAdaptive.m` already handles missing `debug_step` and `debug_run` fields gracefully using `isfield` checks (lines 167-172). The tracking will work without these fields - it just won't include step/run information in the debug file, which is fine since the main tracking is done via `options.initReach_tracking` in `initReach_adaptive.m`.

## Status

✅ **Fixed** - MATLAB should now run without errors. The tracking will still work, just without step/run info in the reduction debug file (which is acceptable).
