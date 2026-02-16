# quadMap Interval Handling Issue - Summary

## Critical Finding

The increasing relative differences in `errorSec` (20.36% → 24.31% → 24.88% → 28.93%) are caused by **how Python handles Interval matrices in `quadMap`**.

## The Problem

### MATLAB Behavior
```matlab
quadMat = Zmat'*Q{i}*Zmat;  % Q{i} is an Interval
% quadMat is an Interval matrix

G(i,1:gens) = 0.5*diag(quadMat(2:gens+1,2:gens+1));
c(i,1) = quadMat(1,1) + sum(G(i,1:gens));
```

**Key Finding**: MATLAB **cannot** directly assign an Interval to a numeric array. This means:
- Either `quadMat` is **not** an Interval at this point (converted earlier)
- Or MATLAB uses a **specific conversion method** (supremum, infimum, or center)

### Python Behavior
```python
quadMat = Zmat.T @ Q_i @ Zmat  # Q_i is an Interval
# quadMat is an Interval

# PROBLEM: Python uses center() which is an approximation!
if isinstance(quadMat, Interval):
    quadMat_center = quadMat.center()  # MIDPOINT APPROXIMATION
    quadMat = np.asarray(quadMat_center)
```

**Python approximates with the midpoint** - it loses interval information.

## Hypothesis

MATLAB must be using one of these methods to convert Interval to numeric:
1. **`supremum()`** - Uses the upper bound (conservative)
2. **`infimum()`** - Uses the lower bound
3. **`center()`** - Uses the midpoint (same as Python, but maybe applied differently)
4. **Some other method** - Applied during matrix operations

## Next Steps

1. **Test MATLAB's actual behavior**: Run `quadMap` with Interval Hessian and check what numeric values are extracted
2. **Compare extraction methods**: Test supremum, infimum, and center to see which matches MATLAB
3. **Fix Python implementation**: Match MATLAB's exact conversion method

## Impact

This fix should:
- Reduce `errorSec` differences from 20-29% to <1%
- Prevent the increasing divergence over steps
- Match MATLAB's behavior exactly

## Test Results

From `test_matlab_interval_extraction.m`:
- MATLAB **cannot** assign Interval directly to numeric array
- `0.5*diag(I)` returns an Interval, not numeric
- There must be an implicit conversion happening in MATLAB's `quadMap`

## Investigation Needed

1. Check if MATLAB's Interval matrix multiplication automatically converts
2. Check if `diag()` on Interval has special behavior
3. Check if there's a `double()` or similar conversion method for Interval
4. Compare actual `quadMat` values between Python and MATLAB
