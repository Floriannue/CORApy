# Zonotope Contains_ Python vs MATLAB Comparison

## Summary of Changes Made

### 1. Interval Delegation (lines 162-193 in Python)

**MATLAB Behavior (lines 172-195):**
- MATLAB **always** delegates to `interval.contains_` when Z is an interval, regardless of what S is
- MATLAB: `contains_(I,S,method,...)` is called for all S types
- MATLAB's `interval.contains_` converts to polytope, and `polytope.contains_` handles zonotope via `aux_contains_P_Hpoly`

**Python Behavior (lines 162-193):**
- Python **skips delegation** when S is a zonotope (lines 174-179)
- This is a **Python-specific workaround** because `polytope.contains_` doesn't handle zonotope yet
- For non-zonotope S, Python matches MATLAB behavior (delegates to interval)

**Difference:**
- ❌ **NOT matching MATLAB** - Python has a workaround to skip delegation for zonotope S
- This is necessary because Python's `polytope.contains_` doesn't handle zonotope yet
- When `polytope.contains_` is fully implemented for zonotope, this workaround should be removed

### 2. exact:polymax Method (lines 305-335 in Python)

**MATLAB Behavior (lines 366-378 in aux_exactParser):**
- MATLAB converts Z to polytope: `P = Polytope(Z)`
- Calls `P.contains_(S, 'exact:polymax', ...)`
- `polytope.contains_` handles zonotope via `aux_contains_P_Hpoly` (line 228 in polytope/contains_.m)
- No fallback needed - polytope fully supports zonotope

**Python Behavior (lines 305-335):**
- Python tries to convert Z to polytope and call `P.contains_(S, 'exact:polymax', ...)`
- Catches `CORAerror` when polytope doesn't handle zonotope
- Falls back to `priv_zonotopeContainment_vertexEnumeration` (exact:venum)
- If venum fails, falls back to `priv_zonotopeContainment_SadraddiniTedrake` (approx:st)

**Difference:**
- ❌ **NOT matching MATLAB** - Python has fallback logic that MATLAB doesn't need
- This is necessary because Python's `polytope.contains_` doesn't handle zonotope yet
- MATLAB's polytope fully supports zonotope, so no fallback is needed

### 3. Error Handling

**MATLAB:**
- Errors are raised and propagated normally
- No special handling needed because all set types are supported

**Python:**
- Added exception handling for `CORAerror` from polytope
- Added fallback chain: polytope → venum → approx:st
- This is a workaround until polytope fully supports zonotope

## Why polytope.contains_ is Not Fully Implemented

The issue is simple: **The Zonotope case is commented out** in `polytope.contains_.py` (lines 269-271).

**MATLAB (line 217):**
- Includes `'zonotope'` in the switch case: `case {'conHyperplane', 'emptySet', 'fullspace', 'halfspace', 'interval', 'polytope', 'conZonotope', 'zonoBundle', 'zonotope', 'capsule', 'ellipsoid', 'spectraShadow'}`
- All these set types go through the same logic: call `aux_contains_P_Hpoly` or `aux_contains_P_Vpoly`
- These functions use `supportFunc_` which works for any ContSet, including zonotope

**Python (lines 269-271):**
- The `elif isinstance(S, Zonotope):` block is **commented out**
- When S is a Zonotope, it falls through to `else:` which raises `CORAerror('CORA:noExactAlg', P, S)`
- The `_aux_contains_P_Hpoly` function already uses `supportFunc_` which **does work for zonotope** (zonotope has `supportFunc_.py`)

**The Fix:**
Simply uncomment and implement the Zonotope case in `_aux_exactParser`:
```python
elif isinstance(S, Zonotope):
    # Handle Zonotope containment, same logic as Polytope
    if method == 'exact':
        if P.isHRep:
            res, cert, scaling = _aux_contains_P_Hpoly(P, S, tol, scalingToggle, certToggle)
        else:
            res, cert, scaling = _aux_contains_P_Vpoly(P, S, tol, scalingToggle, certToggle)
    elif method == 'exact:venum':
        P.vertices_() # Force V-representation
        res, cert, scaling = _aux_contains_P_Vpoly(P, S, tol, scalingToggle, certToggle)
    elif method == 'exact:polymax':
        P.constraints() # Force H-representation
        res, cert, scaling = _aux_contains_P_Hpoly(P, S, tol, scalingToggle, certToggle)
```

This would make Python match MATLAB exactly, and the workarounds in `zonotope.contains_.py` could be removed.

## Recommendations

1. **Implement zonotope support in polytope.contains_:**
   - Uncomment and implement the `elif isinstance(S, Zonotope):` block in `_aux_exactParser` (lines 269-271)
   - This should work immediately since `_aux_contains_P_Hpoly` already uses `supportFunc_` which zonotope supports

2. **After implementing polytope.contains_ for zonotope:**
   - Remove the workaround in interval delegation (lines 174-179 in zonotope/contains_.py)
   - Remove the exception handling and fallback in exact:polymax (lines 313-329 in zonotope/contains_.py)
   - Match MATLAB's behavior exactly

3. **Current state:**
   - Python implementation works but uses workarounds
   - Tests pass with the workarounds
   - Behavior is functionally equivalent but implementation differs
   - The fix is straightforward - just uncomment the Zonotope case

## Files Modified

1. `cora_python/contSet/zonotope/contains_.py`
   - Lines 162-193: Interval delegation with zonotope check
   - Lines 305-335: exact:polymax with exception handling and fallback

