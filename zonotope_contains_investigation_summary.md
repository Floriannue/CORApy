# Zonotope Contains Investigation Summary

## Issues Found and Fixed

### 1. ✅ Fixed: `interval.polytope()` boolean indexing bug
**Problem**: Boolean indexing failed when eliminating unbounded directions.
**Fix**: Properly handle array flattening and reshaping.

### 2. ✅ Fixed: `polytope.minus()` constraint update bug  
**Problem**: When subtracting a vector `q` from polytope `P`, constraints were updated incorrectly.
- **Wrong**: `b_new = b + A*q` 
- **Correct**: `b_new = b - A*q` (for `P - q`)

**Root Cause**: The formula for `P - q` where `P = {x : Ax <= b}` should be:
- `P - q = {x - q : x in P} = {x' : A(x' + q) <= b} = {x' : Ax' <= b - Aq}`
- So `b_new = b - Aq`, not `b + Aq`

**MATLAB Reference**: `priv_plus_minus_vector` computes `P+v` as `b = b + A*v`, so for `P-v` it calls with `-v`, giving `b = b + A*(-v) = b - A*v`.

### 3. ✅ Fixed: `isFullDim` tuple return handling
**Problem**: `isFullDim` returns `(bool, Optional[np.ndarray])` tuple, but code expected boolean.
**Fix**: Extract first element of tuple if it's a tuple.

## Test Results

### Before Fixes
- ❌ `test_outer_zonotope_is_interval` - FAILED
- ❌ `test_zono_in_zono` - FAILED  
- ❌ `test_degenerate_sets` - FAILED (newly discovered)

### After Fixes
- ✅ `test_outer_zonotope_is_interval` - PASSED
- ✅ `test_zono_in_zono` - PASSED
- ⚠️ `test_degenerate_sets` - Still FAILING (needs further investigation)

## Remaining Issue: `test_degenerate_sets`

**Test Case**:
- Z1: Degenerate zonotope at x=5.0 (only generator in y-direction)
- Z2: Full-dimensional zonotope centered at x=5.65

**Expected**: `Z1.contains_(Z2)` should return `False` (Z2's x-range [5.6, 5.7] is outside Z1's x-range [5.0, 5.0])

**Actual**: Returns `True`

**Investigation**:
- Buffering works correctly: Z1 gets buffered with `tol * Interval(-ones, ones)`
- After buffering, Z1_buffered still has x-range [5, 5] (degenerate in x-direction)
- Manual containment check shows x-direction is NOT contained
- But `contains_` returns `True`

**Possible Causes**:
1. Numerical tolerance issues in containment check
2. Issue with how degenerate sets are handled in polytope conversion
3. Issue with support function calculation for degenerate sets

## Comparison with MATLAB

### MATLAB Behavior (`contains_.m`):
1. Checks if Z represents interval → delegates to `interval.contains_`
2. Buffers degenerate sets: `Z = Z + tol*interval(-ones, ones)`
3. For zonotope-zonotope containment, converts to polytope and uses `polytope.contains_`

### Python Implementation:
- ✅ Correctly delegates to interval.contains_ when Z represents interval
- ✅ Buffers degenerate sets correctly
- ✅ Converts to polytope for containment check
- ⚠️ Issue with degenerate set containment check (needs further investigation)

## Files Modified

1. `cora_python/contSet/interval/polytope.py` - Fixed boolean indexing
2. `cora_python/contSet/polytope/minus.py` - Fixed constraint update formula
3. `cora_python/contSet/zonotope/contains_.py` - Fixed `isFullDim` tuple handling

## Next Steps

1. Investigate why `test_degenerate_sets` fails
2. Compare degenerate set handling with MATLAB
3. Check if support function calculation handles degenerate sets correctly
4. Verify polytope conversion for degenerate zonotopes

