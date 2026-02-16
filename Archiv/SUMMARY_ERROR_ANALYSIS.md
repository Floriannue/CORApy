# Summary: error_adm_horizon Growth Analysis

## Test Results

✅ **Python test completed successfully:**
- 897 steps tracked
- Growth from 5.34e-02 to 4.42e+11 (8.27 trillion times)
- First explosion point: Step 451

## Critical Finding: Step 450 → 451 Mismatch

**Mismatch detected:**
- Step 450 final `trueError`: 3.687807e+05
- Step 451 initial `error_adm_horizon`: 1.253572e+06
- **Jump ratio: 3.40x** - NOT from `trueError`!

## Code Analysis

### Where error_adm_horizon is Set

1. **run == 1, i == 1** (line 440):
   ```python
   options['error_adm_horizon'] = trueError
   ```

2. **run == 1, i > 1** (line 452):
   ```python
   options['error_adm_horizon'] = trueError
   ```

3. **run == 2** (line 489):
   ```python
   options['error_adm_Deltatopt'] = trueError
   # Note: error_adm_horizon is NOT set here!
   ```

4. **_aux_nextStepTensorOrder** (lines 603, 619):
   ```python
   options['error_adm_horizon'] = np.zeros((nlnsys.nr_of_dims, 1))
   ```

### The Problem

The trace files only show **run == 1** (inner loop iterations). **run == 2** doesn't have an inner loop, so we don't see its `trueError` in the trace files.

**Hypothesis:** 
- Step 450 run == 1: `error_adm_horizon = trueError` (3.687807e+05) - this is what we see in trace
- Step 450 run == 2: `error_adm_Deltatopt = trueError` (unknown value, possibly 1.253572e+06)
- **If `error_adm_horizon` is being set from `error_adm_Deltatopt` instead of run == 1's `trueError`, that would explain the jump**

## Next Steps

1. ✅ **Python tracking complete** - 897 trace files created
2. ⏳ **MATLAB test** - Need to run MATLAB test to compare
3. ⏳ **Compare Step 451** - Use comparison tool once MATLAB trace is available
4. ⏳ **Add run == 2 tracking** - Track `trueError` in run == 2 to see if it matches Step 451's initial value
5. ⏳ **Check error_adm_Deltatopt** - Verify if `error_adm_horizon` is being set from `error_adm_Deltatopt` in some code path

## Files Created

- `analyze_error_growth.py` - Growth pattern analysis
- `analyze_error_adm_horizon_update.py` - Update logic analysis  
- `compare_step451.py` - Step 450-451 comparison
- `ERROR_ADM_HORIZON_ANALYSIS.md` - Detailed analysis
- `SUMMARY_ERROR_ANALYSIS.md` - This summary

## Key Insight

The 3.4x jump from Step 450 to Step 451 suggests that `error_adm_horizon` is NOT being set from the `trueError` we see in the trace files (which is from run == 1). It might be:
1. Set from run == 2's `trueError` (which we don't track)
2. Set from `error_adm_Deltatopt` 
3. Set from a different step's value
4. A bug in the update logic

**Solution:** Add tracking for run == 2's `trueError` to verify the hypothesis.
