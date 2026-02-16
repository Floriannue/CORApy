# CRITICAL FINDING: `reduce('adaptive')` Implementation Mismatch

## The Problem

**Python's `priv_reduceAdaptive` is NOT implemented!** It's just a placeholder that calls `priv_reduceGirard`, while MATLAB uses a completely different algorithm.

### Python Implementation (WRONG)
```python
def priv_reduceAdaptive(Z: 'Zonotope', order, option: str = 'default') -> Tuple['Zonotope', float, np.ndarray]:
    """Adaptive reduction method - simplified implementation"""
    Z_reduced = priv_reduceGirard(Z, 1)  # ❌ Just calls Girard's method!
    dHerror = 0.0  # Placeholder
    gredIdx = np.array([])  # Placeholder
    return Z_reduced, dHerror, gredIdx
```

### MATLAB Implementation (CORRECT)
```matlab
% Uses 'girard' type by default
norminf = max(Gabs,[],1);               % faster than: vecnorm(G,Inf);
normsum = sum(Gabs,1);                  % faster than: vecnorm(G,1);
[h,idx] = mink(normsum - norminf,nrG);  % ✅ Different sorting criterion!

% Then computes cumulative Hausdorff distance
gensdiag = cumsum(gensred-mugensred,2);
h = 2 * vecnorm(gensdiag,2);
redIdx = find(h <= dHmax,1,'last');     % ✅ Selects based on dHmax threshold
```

## Impact

This is **THE ROOT CAUSE** of the divergence:

1. **Different Generator Selection**:
   - Python: Sorts by generator norm (Girard's method)
   - MATLAB: Sorts by `normsum - norminf` (adaptive method)
   - **Result**: Different generators are selected for reduction

2. **Different Set Representations**:
   - Different generator selection → different reduced sets
   - Even if the original sets are similar, the reduced sets differ
   - This propagates through all subsequent computations

3. **Cascading Effects**:
   - Different reduced sets → different `Z` → different `errorSec` → different `VerrorDyn`
   - This explains why Python's VerrorDyn is 18-27% different from MATLAB's

## Why This Matters

The `reduce('adaptive')` function is called **multiple times per step**:
- `R.reduce('adaptive', redFactor * 5)` in `reach_adaptive` (for Rti)
- `R.reduce('adaptive', redFactor)` in `reach_adaptive` (for Rtp)
- `R.reduce('adaptive', sqrt(redFactor))` in `priv_abstractionError_adaptive` (for Rred)
- `VerrorDyn.reduce('adaptive', 10 * redFactor)` in `priv_abstractionError_adaptive`

Each call with the wrong algorithm causes different generator selection, compounding the differences.

## The Fix

**Implement `priv_reduceAdaptive` correctly** to match MATLAB's algorithm:

1. Compute `norminf = max(Gabs, axis=0)` for each generator
2. Compute `normsum = sum(Gabs, axis=0)` for each generator
3. Sort by `normsum - norminf` using `np.argpartition` (equivalent to MATLAB's `mink`)
4. Compute cumulative Hausdorff distance `h = 2 * vecnorm(gensdiag, 2)`
5. Select generators to reduce based on `dHmax` threshold
6. Return reduced zonotope, `dHerror`, and `gredIdx`

## Expected Impact

Once fixed, we should see:
- ✅ Same generator selection as MATLAB
- ✅ Same reduced set representations
- ✅ Much smaller differences in `VerrorDyn` (should be <1% instead of 18-27%)
- ✅ Python should match MATLAB's behavior much more closely

## Priority

**CRITICAL** - This is the primary source of divergence. Fixing this should significantly reduce the differences between Python and MATLAB.
