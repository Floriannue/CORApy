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

## Key Finding

**MATLAB has NO safeguards** for preventing `error_adm_horizon` from becoming huge. The code is identical in both versions.

When `VerrorDyn` has huge generators:
- Reduction with `10 * redFactor` doesn't help because `dHmax` is proportional to the input size
- `quadMap` amplifies quadratically (expected behavior)
- `trueError` becomes huge
- `error_adm_horizon = trueError` propagates to next step
- Cycle continues

## Conclusion

The unbounded growth is **expected behavior** when the system diverges. MATLAB would also experience this issue. The system should raise `CORA:reachSetExplosion` when `VerrorDyn` or `trueError` becomes infinite, which we've already added.

The issue is that the check happens **after** `errorSec` is computed, but `errorSec` itself can become huge before the check. We should add an earlier check when `Rmax` or `Z` becomes too large.


pytest cora_python/tests/contDynamics/nonlinearSys/test_nonlinearSys_reach_adaptive_01_jetEngine.py::test_nonlinearSys_reach_adaptive_01_jetEngine -v -s 2>&1