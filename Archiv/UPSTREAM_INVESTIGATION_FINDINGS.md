# Upstream Computations Investigation - Key Findings

## Summary

Successfully implemented comprehensive tracking and comparison of upstream computations between Python and MATLAB. The comparison now correctly matches converged iterations by (step, run) pairs.

## Key Findings

### 1. **Steps 1-3: Excellent Agreement**
- VerrorDyn: 0.0024% - 0.18% difference
- Rerror rerr1: 0.0024% - 0.16% difference
- **Conclusion**: Translation is correct for initial steps

### 2. **Steps 4+: Significant Divergence**
- **VerrorDyn before errorSolution**: Python is 18-27% **smaller** than MATLAB
- **Rerror rerr1**: Python is 2-12% **smaller** than MATLAB
- **Impact**: Smaller `rerr1` in Python leads to different time step selection in `_aux_optimaldeltat`

### 3. **Root Cause Analysis**

The divergence starts at Step 4:
- Step 3: 0.18% difference in VerrorDyn
- Step 4: 20.9% difference in VerrorDyn

This suggests:
1. **Accumulated differences** from previous steps compound
2. **Different time step selections** in Steps 1-3 (even though small) lead to different reachable set sizes
3. **Different reachable set sizes** → different `Z` → different `errorSec` → different `VerrorDyn`

### 4. **Why Python's Values Are Smaller**

Python's smaller `VerrorDyn` and `rerr1` values suggest:
- Python's reachable sets (`R`) might be growing faster
- Faster growth → larger `R` → larger `Z` → but wait, Python's `VerrorDyn` is smaller...

Actually, this is counterintuitive. If Python's `R` is larger, then `Z` should be larger, and `errorSec` should be larger. But Python's `VerrorDyn` is smaller.

**Possible explanations**:
1. Python's `errorLagr` (third-order term) is much smaller
2. Python's `quadMap` computation produces smaller results
3. Python's reduction (`reduce('adaptive')`) is more aggressive
4. Different generator selection in reduction leads to different set sizes

### 5. **Impact on Time Step Selection**

The smaller `rerr1` in Python (2-12% smaller) directly impacts `_aux_optimaldeltat`:
- Smaller `rerr1` → different objective function values
- Different objective function → different `bestIdxnew` selection
- Different time step → different reachable set growth
- This compounds over time, leading to early abortion

## Next Steps

1. **Compare `errorSec` and `errorLagr` separately** to identify which component diverges
2. **Compare `Z` before `quadMap`** to see if the input differs
3. **Compare `quadMap` results** directly to verify the computation
4. **Compare reduction behavior** to see if generator selection differs
5. **Compare reachable set sizes (`R`)** to understand why `Z` differs

## Files Modified

- `cora_python/contDynamics/nonlinearSys/linReach_adaptive.py`: Added tracking
- `cora_python/contDynamics/nonlinearSys/private/priv_abstractionError_adaptive.py`: Added tracking
- `cora_matlab/contDynamics/@nonlinearSys/linReach_adaptive.m`: Added tracking
- `cora_matlab/contDynamics/@nonlinearSys/private/priv_abstractionError_adaptive.m`: Added tracking
- `compare_upstream_computations.py`: Comparison script

## Status

✅ Tracking infrastructure complete and working
✅ Comparison script correctly matches converged iterations
✅ Identified divergence point (Step 4)
⚠️ Need to investigate `errorSec`, `errorLagr`, and `Z` to find root cause
