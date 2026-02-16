# JetEngine Differences Analysis

## Summary

Python and MATLAB produce **different results** for the jetEngine adaptive reachability test:

| Metric | MATLAB | Python | Difference |
|--------|--------|--------|------------|
| Number of steps | 237 | 867 | Python takes 3.66x more steps |
| Final time | 8.000000 | 1.800735 | Python stops early (22.5% of expected) |
| Final radius | 5.796034e-02 | 1.202137e+04 | Python's radius is 207,000x larger |
| Algorithm | 'lin' | 'lin' | ✓ Matches |

## Root Cause: Early Abortion

Python aborts early due to the abortion condition in `_aux_checkForAbortion`:

```
if remTime / lastNsteps > 1e9:
    abortAnalysis = True
```

### Why Python Aborts

1. **Time steps become very small**: Python's time steps shrink to ~1e-10
2. **Last 10 steps sum is tiny**: When last 10 steps sum to ~1e-9, and remaining time is ~6.2 seconds, the ratio becomes ~6e9, which is > 1e9
3. **Abortion triggers**: The condition `remTime / lastNsteps > 1e9` becomes true, causing early termination

### Why MATLAB Doesn't Abort

MATLAB's time steps remain reasonable:
- Min: 7.09e-03
- Max: 9.44e-02  
- Mean: 3.39e-02
- Last 10 sum: 0.616

With these time steps, the ratio `remTime / lastNsteps` stays well below 1e9, so MATLAB completes successfully.

## Investigation Needed

The question is: **Why do Python's time steps become so small while MATLAB's don't?**

Both implementations have the same `finitehorizon` bug (line 84 in MATLAB, line 78 in Python):
```matlab
min([params.tFinal - options.t, finitehorizon]);  % Result not assigned!
```

This bug should affect both equally, but Python's time steps shrink much faster. Possible causes:

1. **Numerical precision differences**: Different floating-point rounding
2. **Different BLAS libraries**: MATLAB uses MKL, Python uses OpenBLAS
3. **Order of operations**: Small differences in computation order
4. **Accumulation of errors**: Small differences compound over many steps
5. **Different `varphi` computation**: The `varphi` value used to compute `finitehorizon` might differ

## Test Tolerances

The test now uses **tighter tolerances**:

- **Final time**: 1e-6 absolute tolerance (if completed), 0.1 if aborted early
- **Final radius**: 1e-4 (0.01%) relative tolerance - only checked if Python completes successfully
- **Algorithm**: Must match exactly

## Next Steps

1. **Investigate time step computation**: Compare `finitehorizon`, `varphi`, and related values between MATLAB and Python
2. **Check intermediate values**: Use trace files to compare values at each step
3. **Identify divergence point**: Find the first step where Python and MATLAB diverge significantly
4. **Fix the root cause**: Once identified, fix the numerical difference causing Python's time steps to shrink

## Current Status

- ✅ **Translation is truthful**: Python test matches MATLAB example exactly
- ✅ **Expected values embedded**: MATLAB-generated expected values are in the test
- ⚠️ **Early abortion bug**: Python stops early due to very small time steps
- ⚠️ **Test will fail**: Until the early abortion is fixed, the test will fail on final time assertion
