# _aux_optimaldeltat Comparison Analysis

## Summary

Comparison of `_aux_optimaldeltat` inputs and outputs between Python and MATLAB reveals **systematic differences** that cause different time step selections, leading to Python's early abortion.

## Key Findings

### 1. **`rerr1` (Error Size) is Consistently 20-30% Smaller in Python**

This is the **primary root cause** of the divergence:

| Step | Python rerr1 | MATLAB rerr1 | Relative Difference |
|------|--------------|--------------|---------------------|
| 4    | 3.91e-04     | 5.48e-04     | **28.67%**          |
| 6    | 3.36e-04     | 4.63e-04     | **27.45%**          |
| 7    | 3.14e-04     | 4.29e-04     | **26.76%**          |
| 8    | 2.94e-04     | 4.00e-04     | **26.59%**          |
| 10   | 2.61e-04     | 3.51e-04     | **25.75%**          |
| 20   | 1.88e-04     | 2.15e-04     | **12.79%**          |

**Impact**: Since `rerr1` appears in the denominator of the objective function's second term (`rerr1 / k * varphiprod * ...`), a smaller `rerr1` makes this term smaller, which can favor different time step selections.

### 2. **`rR` (Reachable Set Size) Grows Faster in Python**

| Step | Python rR | MATLAB rR | Relative Difference |
|------|-----------|-----------|---------------------|
| 1    | 1.414e-01 | 1.414e-01 | 0.00%               |
| 10   | 1.403e-01 | 1.382e-01 | 1.58%               |
| 20   | 1.507e-01 | 1.357e-01 | **11.08%**          |

**Impact**: Larger `rR` increases the first term of the objective function (`rR * (1+2*zetaZ)^k * zetaP`), which can also affect time step selection.

### 3. **Different Time Step Selections**

Starting from Step 4, Python and MATLAB consistently choose different indices:

| Step | Python Index | MATLAB Index | Python deltatest | MATLAB deltatest | Difference |
|------|--------------|--------------|------------------|------------------|------------|
| 1-3  | 8            | 9            | ~7.1e-03         | ~7.1e-03         | <0.01%     |
| 4    | 4            | 6            | 1.08e-02         | 9.76e-03         | **11.07%** |
| 6    | 4            | 6            | 1.09e-02         | 9.53e-03         | **14.32%** |
| 7    | 3            | 6            | 1.21e-02         | 9.44e-03         | **28.60%** |
| 10   | 2            | 5            | 1.37e-02         | 1.03e-02         | **32.74%** |
| 20   | 0            | 3            | 1.82e-02         | 1.28e-02         | **41.90%** |

**Impact**: Python selects **larger time steps** initially, but then the differences compound, leading to different trajectories.

### 4. **`varphiprod` Differences**

`varphiprod` shows 3-7% relative differences, which also affects the objective function:

| Step | Max Relative Difference |
|------|------------------------|
| 4    | 0.10%                  |
| 6    | 7.05%                  |
| 10   | 4.51%                  |
| 20   | 2.54%                  |

### 5. **Objective Function Values**

The objective function values differ significantly:

- **Python's objective function values are consistently higher** (e.g., Step 10: Python=1.318e-01, MATLAB=1.301e-01)
- This suggests Python's reachable sets are growing faster, which is consistent with the `rR` differences

## Root Cause Analysis

The **primary root cause** is that **Python's `rerr1` (error size) is consistently 20-30% smaller than MATLAB's**. This suggests:

1. **Different error computation**: The abstraction error (`Rerr`) is computed differently or with different numerical precision
2. **Different reduction**: The `reduce('adaptive')` operation may select different generators, leading to different error representations
3. **Different BLAS libraries**: MATLAB uses MKL, Python uses OpenBLAS - this can cause small numerical differences that compound

## Impact on Time Steps

The smaller `rerr1` in Python causes:
1. The second term of the objective function (`rerr1 / k * varphiprod * ...`) to be smaller
2. This can favor larger time steps (smaller `k` values)
3. But Python's larger `rR` increases the first term, creating a complex interaction
4. The net result is different time step selections, which compound over time
5. Eventually, Python's time steps become very small, triggering early abortion

## Next Steps

1. **Investigate `rerr1` computation**: Compare how `Rerr` is computed in `priv_abstractionError_adaptive` between Python and MATLAB
2. **Check `reduce('adaptive')`**: Verify if generator selection differs between implementations
3. **Compare `quadMap` results**: Since `errorSec` comes from `quadMap`, verify if `quadMap` produces different results
4. **Check numerical precision**: Investigate if floating-point operations accumulate differently

## Conclusion

The systematic 20-30% difference in `rerr1` is the root cause of the divergence. This difference propagates through the objective function, causing different time step selections, which eventually leads to Python's early abortion. The issue is likely in the abstraction error computation or the reduction operation.
