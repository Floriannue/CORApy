# Critical Finding: quadMap Interval Handling

## Problem Identified

The increasing relative differences in `errorSec` (20.36% → 24.31% → 24.88% → 28.93%) are caused by **how Python handles Interval matrices in `quadMap`**.

## Root Cause

### MATLAB Behavior
```matlab
quadMat = Zmat'*Q{i}*Zmat;  % Q{i} is an Interval
% quadMat is an Interval matrix
G(i,1:gens) = 0.5*diag(quadMat(2:gens+1,2:gens+1));  % Direct extraction from Interval
c(i,1) = quadMat(1,1) + sum(G(i,1:gens));  % Direct extraction from Interval
```

**MATLAB preserves interval arithmetic** - it extracts elements directly from the Interval matrix.

### Python Behavior
```python
quadMat = Zmat.T @ Q_i @ Zmat  # Q_i is an Interval
# quadMat is an Interval

# PROBLEM: Python uses center() which is an approximation!
if isinstance(quadMat, Interval):
    quadMat_center = quadMat.center()  # MIDPOINT APPROXIMATION
    quadMat = np.asarray(quadMat_center)

G[i, :gens] = 0.5 * np.diag(quadMat[1:gens+1, 1:gens+1])  # Uses center
c[i, 0] = quadMat[0, 0] + np.sum(G[i, :gens])  # Uses center
```

**Python approximates with the midpoint** - it loses interval information by using `center()`.

## Why This Causes Increasing Differences

1. **Step 1**: Small differences accumulate
2. **Step 3**: Differences become noticeable (20.36%)
3. **Steps 4-6**: Differences compound because:
   - Each step uses the previous step's result
   - The approximation error accumulates
   - The interval width grows, making the center approximation less accurate

## The Fix

Python should **preserve interval arithmetic** like MATLAB does. Instead of using `center()`, we should:

1. **Extract interval bounds** for diagonal elements
2. **Use interval arithmetic** for the center computation
3. **Preserve interval information** throughout the computation

### Proposed Solution

Instead of:
```python
if isinstance(quadMat, Interval):
    quadMat_center = quadMat.center()
    quadMat = np.asarray(quadMat_center)
```

We should:
```python
if isinstance(quadMat, Interval):
    # Extract inf and sup for diagonal elements
    quadMat_inf = quadMat.inf
    quadMat_sup = quadMat.sup
    # Use interval arithmetic for computation
    # This matches MATLAB's behavior
```

However, the challenge is that MATLAB's Interval class might handle matrix indexing differently. We need to check:
1. How MATLAB extracts `diag(quadMat(2:gens+1,2:gens+1))` when `quadMat` is an Interval
2. Whether MATLAB uses inf, sup, or center for this extraction
3. How to match MATLAB's exact behavior

## Next Steps

1. **Investigate MATLAB's Interval matrix indexing**: How does MATLAB extract elements from an Interval matrix?
2. **Compare actual quadMat values**: Track `quadMat` before and after conversion in both Python and MATLAB
3. **Test interval extraction**: Verify how diagonal extraction works with Interval matrices
4. **Implement fix**: Match MATLAB's exact interval handling

## Impact

This fix should:
- Reduce `errorSec` differences from 20-29% to <1%
- Prevent the increasing divergence over steps
- Match MATLAB's behavior exactly
