# Translation Status Summary

**Date:** 2025-01-XX  
**Purpose:** Track what's translated vs. what's missing from `translation_plan_linear_nonlinear_hybrid.md`

---

## Why `test_priv_select.py` is Skipping Tests

The tests are skipped because `priv_select` depends on `linReach`, which has **NotImplementedError** for some dependencies:

### Missing Dependencies in `linReach`:

1. **`initReach_inputDependence`** - Line 114: `raise NotImplementedError("initReach_inputDependence not yet translated")`
   - Used for `nonlinParamSys` with `interval` parameter sets
   - **Status:** NOT TRANSLATED

2. **`linParamSys.initReach`** - Line 119: Comment says needs translation
   - Used for linear parameter systems
   - **Status:** NEEDS VERIFICATION

3. **`linearSys.initReach`** - Used for standard linear systems
   - **Status:** EXISTS (but may have dependencies)

---

## Translation Plan Status

### Phase 1: Foundation (Priority: High)
- [x] `priv_precompStatError` - **EXISTS** (found in codebase)
- [x] `priv_abstrerr_lin` - **EXISTS** (found in codebase)
- [x] `priv_abstrerr_poly` - **EXISTS** (found in codebase)

**Status:** ✅ **COMPLETE** - All Phase 1 functions exist

### Phase 2: Nonlinear Core (Priority: High)
- [x] `nonlinearSys.initReach` - **EXISTS** (file found: `cora_python/contDynamics/nonlinearSys/initReach.py`)
- [x] `nonlinearSys.post` - **EXISTS** (file found: `cora_python/contDynamics/nonlinearSys/post.py`)
- [x] `contDynamics.derivatives` - **EXISTS** (file found: `cora_python/contDynamics/contDynamics/derivatives.py`)
- [ ] `nonlinearSys.initReach_adaptive` - **STATUS UNKNOWN** (optional)

**Status:** ⚠️ **MOSTLY COMPLETE** - Core functions exist, but may have missing dependencies

### Phase 3: Linear Verification (Priority: High)
- [x] `linearSys.priv_verifyRA_supportFunc` - **EXISTS** (found in codebase, but has NotImplementedError for some paths)

**Status:** ⚠️ **PARTIAL** - Main function exists but may have incomplete paths

### Phase 4: Hybrid Foundation (Priority: High)
- [ ] `location.guardIntersect` (dispatcher)
- [ ] `location.guardIntersect_zonoGirard` (priority 1)
- [ ] `location.guardIntersect_nondetGuard` (priority 2)
- [ ] `location.guardIntersect_levelSet` (priority 3)
- [ ] `location.guardIntersect_polytope` (priority 4)
- [ ] `location.guardIntersect_conZonotope` (priority 5, optional)
- [ ] `location.reach`
- [ ] `hybridAutomaton.reach`

**Status:** ❌ **NOT STARTED** - None of these are translated

### Phase 5: Optional (Priority: Low)
- [x] `linearSys.priv_reach_krylov` - **EXISTS** (found in codebase)

**Status:** ✅ **COMPLETE**

---

## Specific Missing Functions Causing Test Skips

### In `linReach.py`:
1. **`initReach_inputDependence`** (Line 114)
   - **Purpose:** Handle input-dependent initial sets for parameter systems
   - **Impact:** Tests for `nonlinParamSys` with interval parameters will fail
   - **Priority:** Medium (only affects parameter systems)

### Other Potential Missing Dependencies:
- Check if `linearSys.initReach` is fully implemented
- Check if `linParamSys.initReach` exists and works
- Check if `nonlinearSys.initReach` has all dependencies

---

## Why Tests Skip Instead of Fail

The test code in `test_priv_select.py` catches `NotImplementedError` and `AttributeError`:

```python
try:
    dimForSplit = priv_select(sys, Rinit, params, options)
    # ... assertions ...
except (NotImplementedError, AttributeError) as e:
    pytest.skip(f"Dependencies not yet translated: {e}")
```

This means:
- If `linReach` raises `NotImplementedError`, the test is skipped
- If any dependency is missing (AttributeError), the test is skipped
- Tests don't fail, they just skip with a message

---

## Recommendations

1. **Fix `initReach_inputDependence`**:
   - Translate `cora_matlab/contDynamics/@nonlinParamSys/initReach_inputDependence.m`
   - Or create a stub that handles the case gracefully

2. **Verify `linReach` dependencies**:
   - Check if `linearSys.initReach` is fully working
   - Check if `linParamSys.initReach` exists and works
   - Fix any missing dependencies

3. **Update test expectations**:
   - For now, skipping is acceptable if dependencies are truly missing
   - Once dependencies are fixed, tests should run and pass

4. **Continue with Priority 1 tests**:
   - Focus on tests that don't require `priv_select` or `linReach`
   - Come back to `priv_select` tests after fixing dependencies

---

## Files with NotImplementedError

Found in:
- `cora_python/contDynamics/contDynamics/linReach.py` (2 instances)
- `cora_python/contDynamics/linearSys/reach.py` (check for instances)
- `cora_python/contDynamics/linearSys/private/priv_verifySTL_kochdumper.py` (check)
- `cora_python/contDynamics/linearSys/private/priv_verifyRA_supportFunc.py` (check)
- `cora_python/contDynamics/contDynamics/symVariables.py` (check)
- `cora_python/contDynamics/linearSys/private/priv_reach_adaptive.py` (check)

---

**Conclusion:** Most core functions from the translation plan exist, but `linReach` has a missing dependency (`initReach_inputDependence`) that causes `priv_select` tests to skip. This is expected behavior until that dependency is translated.
