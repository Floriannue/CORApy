# error_adm_horizon Update Logic Analysis

## Summary

The `error_adm_horizon` update logic has been traced through both Run 1 and Run 2. The key finding is:

**Each step starts with the previous step's Run 1 `error_adm_horizon`, and Run 1 sets `error_adm_horizon` to its final `trueError`.**

## Detailed Flow

### Step 450
- **Initial `error_adm_horizon`**: 6.081008e+05 (from Step 449's Run 1)
- **Run 1**:
  - Final `trueError`: 7.698679e+05
  - Sets `error_adm_horizon` to: 7.698679e+05
- **Run 2**:
  - Final `trueError`: 3.687807e+05
  - Sets `error_adm_Deltatopt` to: 3.687807e+05
  - Does NOT change `error_adm_horizon`
- **Final `error_adm_horizon`**: 7.698679e+05 (from Run 1)

### Step 451
- **Initial `error_adm_horizon`**: 7.698679e+05 (from Step 450's Run 1)
- **Run 1**:
  - Final `trueError`: 1.253572e+06
  - Sets `error_adm_horizon` to: 1.253572e+06
- **Run 2**:
  - Final `trueError`: 3.784520e+05
  - Sets `error_adm_Deltatopt` to: 3.784520e+05
  - Does NOT change `error_adm_horizon`
- **Final `error_adm_horizon`**: 1.253572e+06 (from Run 1)

## Growth Analysis

### Step 450 → Step 451
- Step 450 Run 1 `error_adm_horizon`: 7.698679e+05
- Step 451 Run 1 `error_adm_horizon`: 1.253572e+06
- **Growth factor**: 1.626x

This matches the 1.63x growth factor observed in the error growth analysis.

## Key Observations

1. **`error_adm_horizon` is set from Run 1's `trueError`**, not from Run 2's `error_adm_Deltatopt`.

2. **Run 2's `trueError` is typically smaller** than Run 1's `trueError`:
   - Step 450: Run 1 = 7.698679e+05, Run 2 = 3.687807e+05 (2.09x difference)
   - Step 451: Run 1 = 1.253572e+06, Run 2 = 3.784520e+05 (3.31x difference)

3. **The growth comes from Run 1's `trueError` increasing**, which then becomes the next step's initial `error_adm_horizon`.

4. **Run 2's `error_adm_Deltatopt` is used for the next step's initial `error_adm`** (the per-step error bound), but NOT for `error_adm_horizon`.

## Code Locations

- **Run 1 `error_adm_horizon` update**: `linReach_adaptive.py` lines 442, 464
  - `options['error_adm_horizon'] = trueError`
  
- **Run 2 `error_adm_Deltatopt` update**: `linReach_adaptive.py` line 518
  - `options['error_adm_Deltatopt'] = trueError`

## Next Steps

1. Compare with MATLAB to verify this behavior matches.
2. Investigate why Run 1's `trueError` is growing so rapidly (1.626x per step).
3. Check if MATLAB has any safeguards against this growth that Python is missing.
