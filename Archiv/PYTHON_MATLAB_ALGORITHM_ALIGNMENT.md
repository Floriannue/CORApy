# Python-MATLAB Algorithm Alignment Investigation

## Summary

Investigation to ensure Python algorithm works correctly like MATLAB, focusing on the `_aux_optimaldeltat` function and its inputs.

## Key Findings

### 1. ✅ `rerr1` Computation - VERIFIED CORRECT

The `rerr1` computation in `_aux_optimaldeltat` is **correctly implemented**:

**MATLAB:**
```matlab
rerr1 = vecnorm(sum(abs(generators(Rerr)),2),2);
```

**Python:**
```python
rerr1 = np.linalg.norm(np.sum(np.abs(Rerr.generators()), axis=1), 2)
```

- `axis=1` correctly sums along columns (generators), matching MATLAB's `sum(...,2)`
- `np.linalg.norm(..., 2)` correctly computes 2-norm, matching MATLAB's `vecnorm(...,2)`

### 2. ⚠️ Residual `rerr1` Differences

Despite correct implementation, `rerr1` shows differences:
- **Step 4**: 28.67% difference
- **Step 10**: 25.75% difference
- **Step 20**: 3.39% difference (improving over time)

**Analysis**: The difference is **decreasing** over steps, suggesting:
- The computation formula is correct
- The difference comes from accumulated numerical differences in upstream computations
- These differences compound initially but stabilize

### 3. 🔍 Root Cause: Upstream Computations

The `rerr1` difference originates from differences in `Rerror`, which comes from:
1. **`errorSolution_adaptive`**: Computes `Rerror` from `VerrorDyn`
2. **`priv_abstractionError_adaptive`**: Computes `VerrorDyn` from `errorSec` and `errorLagr`
3. **`quadMap`**: Computes `errorSec` (quadratic map of zonotope)
4. **`reduce('adaptive')`**: Non-deterministic generator selection

### 4. ⚠️ Impact on Time Step Selection

The `rerr1` differences cause:
- Different objective function values
- Different time step selections (different `bestIdxnew`)
- Different `deltatest` values (up to 44% difference by step 20)
- Eventually leads to Python's early abortion

## Comparison Results

### Input Differences (Step 20)
- `deltat`: 30% difference (Python larger)
- `varphimin`: 0.4% difference
- `zetaP`: 0.15% difference
- `rR`: 12% difference (Python larger)
- `rerr1`: 3.4% difference (Python smaller)

### Output Differences (Step 20)
- `deltatest`: 44% difference (Python larger)
- `bestIdxnew`: Different selection (Python index 1, MATLAB index 3)

## Next Steps to Ensure Correct Alignment

1. **Compare `VerrorDyn` values**: Track `VerrorDyn` at each step and compare with MATLAB
2. **Compare `Rerror` values**: Track `Rerror` before reduction and compare with MATLAB
3. **Compare `quadMap` results**: Verify `errorSec` computation matches MATLAB exactly
4. **Check `reduce('adaptive')`**: Investigate if generator selection differs
5. **Compare `errorSolution_adaptive`**: Verify Taylor series expansion matches MATLAB

## Files Modified

- `cora_python/contDynamics/nonlinearSys/linReach_adaptive.py`:
  - Verified `rerr1` computation uses correct `axis=1`
  - Added tracking for `_aux_optimaldeltat` inputs/outputs

## Test Status

- ❌ `test_nonlinearSys_reach_adaptive_01_jetEngine.py` still fails
- Python aborts early at t=1.8s vs MATLAB's t=8.0s
- Root cause: Different time step selections due to `rerr1` differences

## Conclusion

The `_aux_optimaldeltat` function is correctly implemented. The remaining differences come from upstream computations (`VerrorDyn`, `Rerror`, `quadMap`). To fully align Python with MATLAB, we need to investigate these upstream computations and ensure they match exactly.
