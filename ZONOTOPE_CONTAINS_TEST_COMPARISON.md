# Zonotope Contains Tests: MATLAB vs Python Comparison

## Summary

This document compares the MATLAB and Python test coverage for zonotope containment (`contains`/`contains_`).

## Test Files Comparison

### MATLAB Tests
1. **test_zonotope_contains.m** - Basic containment tests
2. **testLong_zonotope_contains.m** - Extended tests with multiple methods and set types
3. **testLong_zonotope_contains_SadraddiniTedrake.m** - Specific tests for Sadraddini-Tedrake method

### Python Tests
1. **test_zonotope_contains_.py** - Basic containment tests (translated from test_zonotope_contains.m)
2. **testLong_zonotope_contains.py** - ❌ **MISSING**
3. **testLong_zonotope_contains_SadraddiniTedrake.py** - ❌ **MISSING**

## Detailed Coverage Analysis

### ✅ test_zonotope_contains.m → test_zonotope_contains_.py

| Test Case | MATLAB | Python | Status |
|-----------|--------|--------|--------|
| Point-in-zonotope (2D) | ✅ | ✅ | Complete |
| Degenerate zonotope | ✅ | ✅ | Complete |
| Almost degenerate zonotope | ✅ | ✅ | Complete |
| Empty zonotope | ⚠️ (commented) | ✅ | Python has it |
| Zono-in-zono (2D) | ✅ | ✅ | Complete |
| Inner zonotope is point | ✅ | ✅ | Complete |
| Approx:st method | ✅ | ✅ | Complete |
| Both zonotopes are points | ✅ | ✅ | Complete |
| Outer zonotope is interval | ✅ | ✅ | Complete |
| Degenerate sets | ✅ | ✅ | Complete |

**Status: ✅ FULLY COVERED**

### ❌ testLong_zonotope_contains.m → **MISSING**

This test file covers:

1. **Zonotope x Parallelotope Containment** - Tests all containment methods:
   - `exact`
   - `exact:venum`
   - `exact:polymax`
   - `opt`
   - `approx`
   - `approx:st`
   - `approx:stDual`
   - `sampling`
   - `sampling:primal`
   - `sampling:dual`

2. **Zonotope x Point** - Tests:
   - Point inside (center)
   - Point outside
   - Array of points (`~all(contains(Z, p_array))`)

3. **checkAllContainments Function** - Comprehensive test with:
   - Multiple set types: `capsule`, `conPolyZono`, `conZonotope`, `interval`, `polytope`, `zonoBundle`, `zonotope`, `ellipsoid`, `taylm`, `polyZonotope`, `spectraShadow`
   - Non-exact sets: `taylm`, `conPolyZono`, `polyZonotope`
   - Additional algorithms with specific set restrictions

**Status: ❌ NOT IMPLEMENTED**

### ❌ testLong_zonotope_contains_SadraddiniTedrake.m → **MISSING**

This test file covers:

1. **Random 3D Zonotopes (Non-zero Center)** - 10 iterations:
   - Creates small zonotope `Z_s` (3D, 5 generators)
   - Creates large zonotope `Z_l` (enlarged by 1.2)
   - Compares `approx:st` method with `exact` method
   - Ensures `approx:st` doesn't give false positives

2. **Random 3D Zonotopes (Zero Center)** - 10 iterations:
   - Same as above but with zero center
   - Tests edge case behavior

**Status: ❌ NOT IMPLEMENTED**

## Missing Functionality

### checkAllContainments Function
The MATLAB test uses a helper function `checkAllContainments` that:
- Tests containment between a zonotope and many other set types
- Tests multiple algorithms (`exact:venum`, `exact:polymax`, `approx:st`, etc.)
- Tests degenerate and empty sets
- Validates certificate outputs

**Location**: `cora_matlab/unitTests/global/functions/helper/contSet/contSet/checkAllContainments.m`

**Python Status**: ❌ Not translated

## Recommendations

1. **Create testLong_zonotope_contains.py**:
   - Translate zonotope x parallelotope tests with all methods
   - Translate zonotope x point array tests
   - Consider translating or adapting `checkAllContainments` function

2. **Create testLong_zonotope_contains_SadraddiniTedrake.py**:
   - Translate random 3D zonotope tests
   - Test both non-zero and zero center cases
   - Compare `approx:st` with `exact` method

3. **Verify Method Coverage**:
   - Ensure all methods (`exact:venum`, `exact:polymax`, `approx:st`, `approx:stDual`, `sampling`, etc.) are properly tested
   - Verify certificate outputs are checked

4. **Set Type Coverage**:
   - Verify containment works with all set types mentioned in MATLAB tests
   - Test degenerate and empty set cases

## Test Execution

To run the existing Python tests:
```bash
pytest cora_python/tests/contSet/zonotope/test_zonotope_contains_.py -v
```

To run MATLAB tests (for comparison):
```matlab
% In MATLAB
test_zonotope_contains
testLong_zonotope_contains
testLong_zonotope_contains_SadraddiniTedrake
```

