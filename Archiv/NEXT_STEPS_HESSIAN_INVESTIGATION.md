# Next Steps: Hessian Investigation

## Current Status

We have identified that the divergence is in `errorSec` (computed by `quadMap`), not in `Z` or `errorLagr`.

## Root Cause Analysis Summary

- **Z differences**: 0.08-1.05% (excellent)
- **errorSec differences**: 20-29% (LARGE - ROOT CAUSE)
- **errorLagr differences**: 0.27-6.35% (good)
- **VerrorDyn differences**: 19-28% (matches errorSec, since VerrorDyn = errorSec + errorLagr)

## Investigation Plan

### 1. Compare H (Hessian) Values ✅ IN PROGRESS

**Status**: Added H tracking to both Python and MATLAB code.

**Next Steps**:
- Re-run MATLAB tracking script (fix any errors)
- Compare H values between Python and MATLAB
- If H differs → issue in Hessian computation
- If H matches but errorSec differs → issue in quadMap computation

### 2. Compare quadMap Implementation

**Key Formula**:
```python
errorSec = 0.5 * Z.quadMap(H)
```

**MATLAB Formula** (from `aux_quadMapSingle`):
```matlab
quadMat = Zmat'*Q{i}*Zmat;
G(i,1:gens) = 0.5*diag(quadMat(2:gens+1,2:gens+1));
c(i,1) = quadMat(1,1) + sum(G(i,1:gens));
```

**Python Formula** (from `_aux_quadMapSingle`):
```python
quadMat = Zmat.T @ Q_i @ Zmat
quadMat_sub = quadMat[1:gens+1, 1:gens+1]
G[i, :gens] = 0.5 * np.diag(quadMat_sub)
c[i, 0] = quadMat[0, 0] + np.sum(G[i, :gens])
```

**Potential Issues**:
1. **Interval handling**: When `Q[i]` (Hessian) is an Interval, Python uses `quadMat.center()` which is an approximation
2. **Matrix multiplication order**: Need to verify `Zmat.T @ Q_i @ Zmat` matches `Zmat'*Q{i}*Zmat`
3. **Indexing**: MATLAB uses 1-based indexing, Python uses 0-based - need to verify all indexing is correct

### 3. Test quadMap Directly

Create a test that:
- Uses the same Z and H from Step 3 (where divergence starts)
- Calls `quadMap` directly in both Python and MATLAB
- Compares results

### 4. Investigate Numerical Precision

- Check if BLAS differences affect `quadMap`
- Consider using MKL for Python
- Verify order of operations matches MATLAB

## Files Modified

1. `cora_python/contDynamics/nonlinearSys/private/priv_abstractionError_adaptive.py`:
   - Added H tracking before quadMap call

2. `cora_matlab/contDynamics/@nonlinearSys/private/priv_abstractionError_adaptive.m`:
   - Added H tracking before quadMap call

3. `compare_hessian_values.py`:
   - Script to compare H values between Python and MATLAB

## Expected Outcomes

1. **If H differs**: The issue is in Hessian computation (likely in `nlnsys.hessian()`)
2. **If H matches**: The issue is in `quadMap` computation (likely in interval handling or matrix multiplication)
