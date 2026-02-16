# Comparison: error_adm_horizon Growth Issue

## Root Cause Analysis

### The Feedback Loop

1. **Step 348**: `error_adm_horizon` becomes huge (from previous step)
2. **Step 349**: Starts with `error_adm_horizon max = 9.484617e+75`
3. `Verror = Zonotope(zeros, diag(error_adm))` has huge generators
4. `errorSolution_adaptive` → `RallError radius = 1.961158e+39`
5. `Rmax = Rlinti + RallError` → `Rmax radius ≈ 1.961158e+39`
6. `Z = Rred.cartProd_(U)` → `Z radius = 1.961158e+39` (reduction doesn't help)
7. `errorSec = 0.5 * Z.quadMap(H)` → **quadratic amplification** → `VerrorDyn radius = 4.742308e+75`
8. `trueError = abs(center(VerrorDyn)) + sum(abs(generators(VerrorDyn)))` → `trueError` is huge
9. `error_adm_horizon = trueError` (line 336, 348) → next step starts with huge `error_adm`
10. Cycle repeats and grows exponentially

## Code Comparison

### 1. Setting error_adm_horizon

**MATLAB** (`linReach_adaptive.m` lines 332, 348):
```matlab
options.error_adm_horizon = trueError;
```

**Python** (`linReach_adaptive.py` lines 336, 348):
```python
options['error_adm_horizon'] = trueError
```

**Status**: ✅ **IDENTICAL** - No bounds checking in either version

### 2. Computing trueError

**MATLAB** (`priv_abstractionError_adaptive.m` line 203):
```matlab
err = abs(center(VerrorDyn)) + sum(abs(generators(VerrorDyn)),2);
```

**Python** (`priv_abstractionError_adaptive.py` line 285):
```python
err = np.abs(VerrorDyn_center) + np.sum(np.abs(VerrorDyn.generators()), axis=1).reshape(-1, 1)
```

**Status**: ✅ **IDENTICAL** - Same computation

### 3. VerrorDyn Reduction

**MATLAB** (`priv_abstractionError_adaptive.m` line 194, 198):
```matlab
VerrorDyn = reduce(VerrorDyn,'adaptive',10*options.redFactor);
```

**Python** (`priv_abstractionError_adaptive.py` line 263):
```python
VerrorDyn_res = VerrorDyn.reduce('adaptive', 10 * options['redFactor'])
```

**Status**: ✅ **IDENTICAL** - Same reduction factor

### 4. Reduction Logic (priv_reduceAdaptive)

**MATLAB** (`priv_reduceAdaptive.m` line 100):
```matlab
redUntil = find(hext <= dHmax,1,'last');
```

**Python** (`priv_reduceAdaptive.py` lines 173-177):
```python
redUntil_idx = np.where(hext <= dHmax)[0]
if redUntil_idx.size > 0:
    redUntil = redUntil_idx[-1] + 1  # +1 because MATLAB find returns 1-based
else:
    redUntil = 0
```

**Issue**: When `redUntil` is empty in MATLAB, `idxall(1:redUntil)` is empty, so no reduction happens.
In Python, when `redUntil = 0`, `idxall[:0]` is also empty, so behavior matches.

**Status**: ✅ **IDENTICAL** - When input is huge, `dHmax` is huge, so reduction allows huge Hausdorff distance

### 5. quadMap Implementation

**MATLAB** (`quadMap.m` line 127):
```matlab
quadMat = Gext'*Q{i}*Gext;
```

**Python** (`quadMap.py` line 143):
```python
quadMat = Gext.T @ Q[i] @ Gext
```

**Status**: ✅ **IDENTICAL** - Quadratic amplification is expected behavior

### 6. error_adm Update in Inner Loop

**MATLAB** (`linReach_adaptive.m` line 243):
```matlab
error_adm = 1.1 * trueError;
```

**Python** (`linReach_adaptive.py` line 243):
```python
error_adm = 1.1 * trueError
```

**Status**: ✅ **IDENTICAL** - No bounds checking

### 7. perfIndCurr Computation and Convergence Check

**MATLAB** (`linReach_adaptive.m` lines 229-240):
```matlab
perfIndCurr = max(trueError ./ error_adm);
if perfIndCurr <= 1 || ~any(trueError)
    perfInds(perfIndCounter) = perfIndCurr;
    options.Lconverged = true;
    break
elseif perfIndCounter > 1
    perfInds(perfIndCounter) = perfIndCurr;
    if perfIndCounter > 2 && perfInds(perfIndCounter) > perfInds(perfIndCounter-1)
        options.Lconverged = false; break
    end
end
```

**Python** (`linReach_adaptive.py` lines 209-242):
```python
with np.errstate(divide='ignore', invalid='ignore'):
    perfIndCurr_ratio = trueError / error_adm
    perfIndCurr = np.max(perfIndCurr_ratio)
    if np.isnan(perfIndCurr):
        perfIndCurr = 0

if perfIndCurr <= 1 or not np.any(trueError):
    perfInds.append(perfIndCurr)
    options['Lconverged'] = True
    break
elif perfIndCounter > 1:
    perfInds.append(perfIndCurr)
    if perfIndCounter > 2 and perfInds[-1] > perfInds[-2]:
        options['Lconverged'] = False
        break
```

**Key Difference**: 
- **MATLAB**: Does NOT handle Inf/NaN explicitly. If `perfIndCurr` is Inf, `perfIndCurr <= 1` is false, loop continues. If NaN, comparison is false, loop continues.
- **Python**: Handles NaN explicitly (sets to 0), but Inf is NOT handled. If `perfIndCurr` is Inf, `perfIndCurr <= 1` is false, loop continues.

**Status**: ⚠️ **MOSTLY IDENTICAL** - Both allow Inf to propagate, but Python handles NaN differently

## Key Finding

**MATLAB has NO safeguards** for preventing `error_adm_horizon` from becoming huge. The code is identical in both versions.

When `VerrorDyn` has huge generators:
- Reduction with `10 * redFactor` doesn't help because `dHmax` is proportional to the input size
- `quadMap` amplifies quadratically (expected behavior)
- `trueError` becomes huge
- `error_adm_horizon = trueError` propagates to next step
- Cycle continues

**MATLAB's approach**:
1. Let computation proceed naturally
2. If `perfIndCurr > 1`, increase `error_adm = 1.1 * trueError`
3. If `perfIndCurr` keeps increasing (diverging), break and halve time step (line 237-238)
4. If computation produces Inf/NaN, it will naturally fail or be caught by divergence check

## Conclusion

The unbounded growth is **expected behavior** when the system diverges. MATLAB would also experience this issue. 

**MATLAB relies on**:
1. **Divergence check**: `if perfIndCounter > 2 && perfInds(perfIndCounter) > perfInds(perfIndCounter-1)` - detects when `perfIndCurr` is increasing
2. **Natural numerical failure**: Inf/NaN will eventually cause operations to fail
3. **Time step halving**: If inner loop doesn't converge, halve time step and reset `error_adm`

**The issue**: The divergence check only triggers if `perfIndCounter > 2` AND `perfInds` is increasing. If `perfIndCurr` is always huge but not increasing, the check won't trigger.

**Solution**: We should match MATLAB's behavior exactly - no early checks, rely on divergence check and natural numerical behavior. The divergence check should catch the issue, but we need to verify it's working correctly.

## Analysis: How MATLAB Handles This

**MATLAB's strategy** (no early checks):
1. **Let computation proceed naturally** - no bounds checking
2. **Divergence detection**: `if perfIndCounter > 2 && perfInds(perfIndCounter) > perfInds(perfIndCounter-1)` 
   - Only triggers if `perfIndCurr` is **increasing**
   - If `perfIndCurr` is Inf and stays Inf, `Inf > Inf` is false, check won't trigger
3. **Natural numerical failure**: Eventually Inf/NaN will cause operations to fail
4. **Time step halving**: If inner loop doesn't converge (`~options.Lconverged`), halve time step and reset `error_adm`

**The Problem**: 
- If `perfIndCurr` becomes huge but **not increasing** (e.g., always Inf), the divergence check won't trigger
- The inner loop will continue indefinitely until a numerical operation fails
- MATLAB would experience the same issue

## Python vs MATLAB: Key Differences

1. **NaN handling**: Python converts NaN to 0 (line 216), MATLAB doesn't handle NaN explicitly
   - **Impact**: If both `trueError` and `error_adm` are zero, Python treats as converged, MATLAB would have NaN comparison
   - **Fix needed**: Match MATLAB - don't convert NaN to 0, let it propagate naturally

2. **Inf handling**: Both allow Inf to propagate, but Python's `np.max()` might handle Inf differently than MATLAB's `max()`
   - **Need to verify**: Does `np.max([Inf, 5])` = Inf like MATLAB `max([Inf, 5])` = Inf?

3. **Divergence check**: Logic appears identical, but need to verify behavior with Inf values

## MATLAB max() Behavior (Verified)

**Test Results**:
- `max([Inf, 5, 3])` = `Inf` ✓
- `max([NaN, 5, 3])` = `5` (NaN is **ignored** if other valid numbers exist)
- `max([Inf, NaN, 5])` = `Inf` (Inf takes precedence)
- `max([0, 2, 3] ./ [0, 1, 1])` = `3` (0/0 = NaN, but max ignores NaN)
- `Inf <= 1` = `false` (0)
- `NaN <= 1` = `false` (0)
- `Inf > Inf` = `false` (0)
- `NaN > NaN` = `false` (0)

**Key Insight**: MATLAB's `max()` **ignores NaN** if there are other valid numbers. This means:
- If `trueError ./ error_adm` contains both NaN and valid numbers, `max()` returns the max of valid numbers
- If all values are NaN, `max()` returns NaN
- Python's `np.max()` has the same behavior (ignores NaN by default)

## Fixes Applied

1. ✅ **Removed NaN to 0 conversion** - Now matches MATLAB's behavior exactly
2. ✅ **Fixed NaN handling in max()** - Changed from `np.max()` to `np.nanmax()` to match MATLAB
   - MATLAB's `max()` ignores NaN if other valid numbers exist
   - NumPy's `np.max()` returns NaN if any element is NaN (different!)
   - `np.nanmax()` ignores NaN like MATLAB's `max()`
3. ✅ **Verified Inf handling** - `np.nanmax()` behaves like MATLAB `max()` with Inf
4. ⏳ **Add debug logging** - To track intermediate values in the inner loop
5. ⏳ **Test with actual jetEngine case** - To see where MATLAB vs Python diverge

## Summary of Changes

### Fixed Issues

1. **NaN handling in max()**: Changed from `np.max()` to `np.nanmax()` to match MATLAB
   - MATLAB's `max()` ignores NaN if other valid numbers exist
   - NumPy's `np.max()` returns NaN if any element is NaN (different behavior!)
   - `np.nanmax()` now correctly matches MATLAB's behavior

2. **Removed NaN to 0 conversion**: MATLAB doesn't convert NaN to 0, so Python shouldn't either
   - NaN <= 1 is false in MATLAB, so loop continues (correct behavior)
   - If both error_adm and trueError are zero, NaN is produced and loop continues

### Verification

- ✅ `np.nanmax([NaN, 5, 3])` = `5` (matches MATLAB)
- ✅ `np.nanmax([Inf, NaN, 5])` = `Inf` (matches MATLAB)
- ✅ `np.nanmax([NaN, NaN])` = `NaN` (matches MATLAB)
- ✅ `Inf <= 1` = `False` (matches MATLAB)
- ✅ `NaN <= 1` = `False` (matches MATLAB)

## Next Steps

1. ✅ **Add comprehensive intermediate value tracking** - Log all key values at each inner loop iteration
   - Added `traceIntermediateValues` option to enable detailed logging
   - Tracks: error_adm, RallError, Rmax, Z, errorSec, VerrorDyn, trueError, perfIndCurr, perfInds
   - Logs to `intermediate_values_step{N}_inner_loop.txt` files

2. ✅ **Create comparison script** - `compare_intermediate_values.py` created
   - Compares MATLAB and Python trace files
   - Reports matches, mismatches, and missing values
   - Configurable tolerance for numerical comparisons

3. ✅ **Test infrastructure ready** - Tracking system implemented and tested
   - ✅ Tracking code added to `linReach_adaptive.py` and `priv_abstractionError_adaptive.py`
   - ✅ Test script `test_tracking_jetEngine.py` created and tested
   - ✅ Comparison script `compare_intermediate_values.py` ready
   - ✅ Fixed tensor indexing bug: `T[i, ind[i][j]]` → `T[i][ind[i][j]]`
   - ✅ Verified tracking creates trace files successfully (53 files created in test)
   - ✅ **MATLAB tracking implemented** - Equivalent tracking code added to MATLAB files
   - ✅ MATLAB test script `test_tracking_jetEngine_matlab.m` created
   - ⏳ **Next**: Run with actual jetEngine case that shows error_adm_horizon growth (use longer tFinal)
   - ⏳ **Next**: Compare MATLAB and Python traces step by step to find divergence point

4. ✅ **Divergence check tracking** - perfInds array and divergence logic now tracked
   - Tracking includes perfInds array and divergence check logic
   - Can verify if divergence is detected correctly when comparing traces

## Usage

**To enable intermediate value tracking in Python:**
```python
options['traceIntermediateValues'] = True
# Run reach_adaptive or linReach_adaptive
# Files will be created: intermediate_values_step{N}_inner_loop.txt
```

**To compare MATLAB and Python traces:**
```bash
python compare_intermediate_values.py matlab_trace.txt python_trace.txt 1e-10
```

**See `README_INTERMEDIATE_VALUE_TRACKING.md` for detailed usage instructions.**
