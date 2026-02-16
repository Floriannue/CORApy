# error_adm_horizon Update Logic Analysis

## Critical Finding

**Step 450 → Step 451 Mismatch:**
- Step 450 final `trueError`: 3.687807e+05
- Step 451 initial `error_adm_horizon`: 1.253572e+06
- **Jump ratio: 3.40x** - This is NOT from `trueError`!

## Code Locations Where error_adm_horizon is Set

### Python (`linReach_adaptive.py`)

1. **Line 440** (run == 1, i == 1):
   ```python
   options['error_adm_horizon'] = trueError
   error_adm = np.zeros((nlnsys.nr_of_dims, 1))
   ```

2. **Line 452** (run == 1, i > 1):
   ```python
   options['error_adm_horizon'] = trueError
   error_adm = options['error_adm_Deltatopt']
   ```

3. **Line 489** (run == 2):
   ```python
   options['error_adm_Deltatopt'] = trueError
   # Note: error_adm_horizon is NOT set here in run == 2
   ```

4. **Line 603, 619** (`_aux_nextStepTensorOrder`):
   ```python
   options['error_adm_horizon'] = np.zeros((nlnsys.nr_of_dims, 1))
   ```

### MATLAB (`linReach_adaptive.m`)

1. **Line 332** (run == 1, i == 1):
   ```matlab
   options.error_adm_horizon = trueError;
   error_adm = zeros(nlnsys.nrOfDims,1);
   ```

2. **Line 348** (run == 1, i > 1):
   ```matlab
   options.error_adm_horizon = trueError;
   error_adm = options.error_adm_Deltatopt;
   ```

3. **Line 366** (run == 2):
   ```matlab
   options.error_adm_Deltatopt = trueError;
   % Note: error_adm_horizon is NOT set here in run == 2
   ```

## The Problem

**Observation from Step 450-451:**
- Step 450 has 2 runs (run == 1 and run == 2)
- In run == 1: `error_adm_horizon = trueError` (line 452)
- In run == 2: `error_adm_Deltatopt = trueError` (line 489), but `error_adm_horizon` is NOT updated
- **Step 451 starts with `error_adm_horizon = 1.253572e+06`, which is 3.4x larger than Step 450's final `trueError`**

## Hypothesis

The `error_adm_horizon` for Step 451 might be coming from:
1. **Step 450's `error_adm_Deltatopt`** - but this should equal `trueError` from run == 2
2. **Step 450's `error_adm` from inner loop** - but this is only used within the step
3. **A different step's value** - possibly from an earlier step that wasn't properly reset
4. **The value is being set incorrectly** - maybe from `error_adm` instead of `trueError`

## Next Steps

1. **Check Step 450's run == 2 trueError** - What was the `trueError` in run == 2?
2. **Check if error_adm_horizon is being modified elsewhere** - Search for all assignments
3. **Compare with MATLAB** - Does MATLAB show the same jump?
4. **Trace the exact flow** - Add more detailed logging to see which code path is taken

## Update Formula

**Expected:**
- Inner loop: `error_adm = 1.1 * trueError` (line 243/350)
- After inner loop (run == 1): `error_adm_horizon = trueError` (line 440/452)
- After inner loop (run == 2): `error_adm_Deltatopt = trueError` (line 489)
- Next step: `error_adm = error_adm_horizon` (line 40)

**Actual (from traces):**
- Step 450 final `trueError`: 3.687807e+05
- Step 451 initial `error_adm_horizon`: 1.253572e+06
- **Mismatch: 3.4x larger than expected**

## Key Questions

1. Is `error_adm_horizon` being set from `error_adm` instead of `trueError`?
2. Is there a code path where `error_adm_horizon` is set to `1.1 * trueError` instead of `trueError`?
3. Is `error_adm_horizon` being set from `error_adm_Deltatopt` in some cases?
4. Is there a bug where the value from a previous step is being reused?
