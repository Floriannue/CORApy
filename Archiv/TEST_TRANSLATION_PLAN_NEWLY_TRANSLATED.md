# Test Translation Plan for Newly Translated Code

**Created:** 2025-01-XX  
**Status:** Planning Phase  
**Priority:** High

---

## Overview

This document provides a comprehensive plan for translating or generating tests for all newly translated code. Tests are categorized as:
- **EXISTING**: MATLAB test file exists and needs translation
- **GENERATED**: No MATLAB test exists, test must be generated based on MATLAB logic

---

## 1. KRYLOV SUBSPACE METHODS

### 1.1 `priv_reach_krylov` + Dependencies

**Status:** Need to check for existing tests or generate

**Files to Test:**
- `cora_python/contDynamics/linearSys/private/priv_reach_krylov.py`
- `cora_python/contDynamics/linearSys/private/priv_initReach_Krylov.py`
- `cora_python/contDynamics/linearSys/private/priv_exponential_Krylov_projected_linSysInput.py`
- `cora_python/contDynamics/linearSys/private/priv_subspace_Krylov_jaweckiBound.py`
- `cora_python/contDynamics/linearSys/private/priv_subspace_Krylov_individual_Jawecki.py`
- `cora_python/contDynamics/linearSys/private/priv_krylov_R_uTrans.py`
- `cora_python/contDynamics/linearSys/private/priv_inputSolution_Krylov.py`
- `cora_python/g/functions/helper/dynamics/contDynamics/linearSys/arnoldi.py`

**Action Plan:**
1. **EXISTING MATLAB tests found:**
   - `cora_matlab/examples/ARCHcompetition/linear/benchmark_linear_reach_ARCH23_heat3D_HEAT03.m`
     - Uses `options.linAlg = 'krylov'`
     - Tests Krylov reachability on heat3D system
     - **Translate:** `test_linearSys_reach_krylov_01_heat3D.py`

2. **Additional tests needed:**
   - Test basic Krylov reachability with small system
   - Test with input sets
   - Test with specifications
   - Test error handling

3. **GENERATED tests for helper functions:**
   - Create: `test_linearSys_priv_initReach_Krylov_01_basic.py` (GENERATED TEST)
   - Create: `test_linearSys_arnoldi_01_basic.py` (GENERATED TEST)
   - Create: `test_linearSys_priv_subspace_Krylov_individual_Jawecki_01_basic.py` (GENERATED TEST)
   - Create: `test_linearSys_priv_krylov_R_uTrans_01_basic.py` (GENERATED TEST)
   - Create: `test_linearSys_priv_inputSolution_Krylov_01_basic.py` (GENERATED TEST)
   - Test based on MATLAB source code logic:
     - Arnoldi iteration produces orthonormal basis
     - Krylov subspace dimension matches expected
     - Happy breakdown detection
     - Error bounds computed correctly
     - Input solution computation

**Test Structure:**
```python
# GENERATED TEST - No MATLAB test file exists
# This test is generated based on MATLAB source code logic
# Source: cora_matlab/contDynamics/@linearSys/private/priv_reach_krylov.m

def test_priv_reach_krylov_basic():
    # Test basic Krylov reachability
    # 1. Create simple linear system
    # 2. Set options.linAlg = 'krylov'
    # 3. Call reach()
    # 4. Verify output structure matches MATLAB
    # 5. Verify reachable sets computed
    pass
```

---

## 2. NONLINEAR SYS: initReach_adaptive

**Status:** Need to check for existing tests

**File to Test:**
- `cora_python/contDynamics/nonlinearSys/initReach_adaptive.py`

**Action Plan:**
1. **EXISTING MATLAB tests found:**
   - `cora_matlab/examples/contDynamics/nonlinearSys/example_nonlinear_reach_12_adaptive.m`
     - Uses `options.alg = 'lin-adaptive'`
     - Tests adaptive reachability on jetEngine system
     - **Translate:** `test_nonlinearSys_reach_adaptive_01_jetEngine.py`
   - `cora_matlab/examples/contDynamics/nonlinearSys/example_nonlinear_reach_13_adaptiveHSCC.m`
     - Multiple systems with adaptive algorithms
     - **Translate:** `test_nonlinearSys_reach_adaptive_02_HSCC.py`
   - `cora_matlab/examples/contDynamics/nonlinearSys/example_nonlinear_reach_14_adaptiveHSCC2.m`
     - More adaptive examples
     - **Translate:** `test_nonlinearSys_reach_adaptive_03_HSCC2.py`

2. **Direct initReach_adaptive test (GENERATED):**
   - Create: `test_nonlinearSys_initReach_adaptive_01_basic.py` (GENERATED TEST)
   - Test based on MATLAB source code logic:
     - Wrapper calls `linReach_adaptive`
     - Returns correct structure (Rnext, options)
     - Options updated correctly
     - Rnext contains 'tp', 'ti', 'R0' keys

---

## 3. NONLINEAR SYS: nonlinearSys Class Constructor

**Status:** Need to check for existing tests

**File to Test:**
- `cora_python/contDynamics/nonlinearSys/nonlinearSys.py`

**Action Plan:**
1. **Search for existing MATLAB tests:**
   - Check if constructor is tested in existing nonlinearSys tests
   - Check `cora_matlab/tests/contDynamics/nonlinearSys/`

2. **If tests exist:**
   - Translate existing constructor tests

3. **If no tests exist (GENERATED):**
   - Create: `test_nonlinearSys_constructor_01_basic.py` (GENERATED TEST)
   - Test based on MATLAB source code logic:
     - Empty constructor
     - Constructor with name, fun, states, inputs
     - Constructor with output function
     - Copy constructor
     - Input validation
     - Property computation

---

## 4. LINEAR SYS: taylorMatrices

**Status:** Need to check for existing tests

**File to Test:**
- `cora_python/contDynamics/linearSys/taylorMatrices.py`

**Action Plan:**
1. **Search for existing MATLAB tests:**
   - Check if tested indirectly in reach tests
   - Check `cora_matlab/tests/contDynamics/linearSys/`

2. **If tests exist:**
   - Translate existing tests

3. **If no tests exist (GENERATED):**
   - Create: `test_linearSys_taylorMatrices_01_basic.py` (GENERATED TEST)
   - Test based on MATLAB source code logic:
     - Computes E (remainder matrix)
     - Computes F (state correction matrix)
     - Computes G (input correction matrix)
     - TaylorLinSys object created if needed
     - Matrices match MATLAB computation

---

## 5. LINEAR SYS: verify

**Status:** Partially exists

**File to Test:**
- `cora_python/contDynamics/linearSys/verify.py`

**Existing Tests:**
- `test_verifyFast_*.py` (multiple files) - These test `verifyFast` which calls `verify`

**Action Plan:**
1. **EXISTING MATLAB tests found:**
   - `cora_matlab/examples/contDynamics/linearSys/example_linear_verify.m`
     - Tests `verify` with `reachavoid:zonotope`
     - Tests with safe/unsafe sets
     - **Translate:** `test_linearSys_verify_01_basic.py`
   - `cora_matlab/examples/contDynamics/linearSys/example_linear_verify_temporalLogic.m`
     - Tests temporal logic verification
     - **Translate:** `test_linearSys_verify_02_temporalLogic.py`
   - `cora_matlab/examples/contDynamics/linearSys/example_linear_verify_electricCircuit.m`
     - Tests on electric circuit system
     - **Translate:** `test_linearSys_verify_03_electricCircuit.py`

2. **Additional tests needed:**
   - Test with `reachavoid:supportFunc` (used by verifyFast)
   - Test error handling for invalid algorithms
   - Test with different specification types

---

## 6. LINEAR SYS & CONTDYNAMICS: outputSet

**Status:** Need to check for existing tests

**Files to Test:**
- `cora_python/contDynamics/linearSys/outputSet.py`
- `cora_python/contDynamics/contDynamics/outputSet.py`

**Action Plan:**
1. **Search for existing MATLAB tests:**
   - Check if tested indirectly in reach tests
   - Check `cora_matlab/tests/contDynamics/`

2. **If tests exist:**
   - Translate existing tests

3. **If no tests exist (GENERATED):**
   - Create: `test_linearSys_outputSet_01_basic.py` (GENERATED TEST)
   - Create: `test_contDynamics_outputSet_01_basic.py` (GENERATED TEST)
   - Test based on MATLAB source code logic:
     - Linear: `Y = C*R + D*U + k + F*V`
     - Nonlinear: Uses tensor order
     - Verror computation
     - Order reduction when specified
     - Empty output equation handling

---

## 7. HYBRID DYNAMICS: location.reach

**Status:** Need to check for existing tests

**File to Test:**
- `cora_python/hybridDynamics/location/reach.py`

**Action Plan:**
1. **Search for existing MATLAB tests:**
   - Check `cora_matlab/examples/hybridDynamics/hybridAutomaton/example_hybrid_reach_*.m`
   - These examples use `hybridAutomaton.reach` which calls `location.reach`

2. **If tests exist:**
   - Extract `location.reach` specific tests from hybrid examples
   - Translate: `test_location_reach_01_basic.py`

3. **If no tests exist (GENERATED):**
   - Create: `test_location_reach_01_basic.py` (GENERATED TEST)
   - Test based on MATLAB source code logic:
     - Computes continuous reachable set
     - Finds guard intersections
     - Computes resets
     - Handles specifications
     - Returns correct structure (R, Rjump, res)

---

## 8. HYBRID DYNAMICS: hybridAutomaton.reach

**Status:** Need to check for existing tests

**File to Test:**
- `cora_python/hybridDynamics/hybridAutomaton/reach.py`

**Action Plan:**
1. **Search for existing MATLAB tests:**
   - Check `cora_matlab/examples/hybridDynamics/hybridAutomaton/example_hybrid_reach_01_bouncingBall.m`
   - Check `cora_matlab/examples/hybridDynamics/hybridAutomaton/example_hybrid_reach_02_powerTrain.m`
   - Check `cora_matlab/examples/hybridDynamics/hybridAutomaton/example_hybrid_reach_03_PLLnoSat.m`
   - Check `cora_matlab/examples/hybridDynamics/hybridAutomaton/example_hybrid_reach_04_spacecraft.m`
   - Check `cora_matlab/examples/hybridDynamics/hybridAutomaton/example_hybrid_reach_05_roomHeating.m`
   - Check `cora_matlab/examples/hybridDynamics/hybridAutomaton/example_hybrid_reach_06_bouncingBallSineWave.m`

2. **If tests exist:**
   - Translate: `test_hybridAutomaton_reach_01_bouncingBall.py`
   - Translate: `test_hybridAutomaton_reach_02_powerTrain.py`
   - Translate: `test_hybridAutomaton_reach_03_PLLnoSat.py`
   - Translate: `test_hybridAutomaton_reach_04_spacecraft.py`
   - Translate: `test_hybridAutomaton_reach_05_roomHeating.py`
   - Translate: `test_hybridAutomaton_reach_06_bouncingBallSineWave.py`

3. **If no tests exist (GENERATED):**
   - Create: `test_hybridAutomaton_reach_01_basic.py` (GENERATED TEST)
   - Test based on MATLAB source code logic:
     - Initializes reachable set queue
     - Handles instant transitions
     - Calls location.reach for each location
     - Processes guard intersections
     - Computes output sets
     - Returns correct structure

---

## 9. HYBRID DYNAMICS: location.guardIntersect Methods

**Status:** Need to check for existing tests

**Files to Test:**
- `cora_python/hybridDynamics/location/guardIntersect.py` (dispatcher)
- `cora_python/hybridDynamics/location/guardIntersect_zonoGirard.py`
- `cora_python/hybridDynamics/location/guardIntersect_nondetGuard.py`
- `cora_python/hybridDynamics/location/guardIntersect_levelSet.py`
- `cora_python/hybridDynamics/location/guardIntersect_polytope.py`
- `cora_python/hybridDynamics/location/guardIntersect_conZonotope.py`
- `cora_python/hybridDynamics/location/guardIntersect_hyperplaneMap.py`
- `cora_python/hybridDynamics/location/guardIntersect_pancake.py`

**Action Plan:**
1. **Search for existing MATLAB tests:**
   - Check `cora_matlab/examples/specification/stl/example_stl_*.m` (some use guardIntersect options)
   - Check `cora_matlab/tests/hybridDynamics/` if exists

2. **If tests exist:**
   - Translate existing tests

3. **If no tests exist (GENERATED):**
   - Create: `test_location_guardIntersect_01_dispatcher.py` (GENERATED TEST)
   - Create: `test_location_guardIntersect_zonoGirard_01_basic.py` (GENERATED TEST)
   - Create: `test_location_guardIntersect_nondetGuard_01_basic.py` (GENERATED TEST)
   - Create: `test_location_guardIntersect_levelSet_01_basic.py` (GENERATED TEST)
   - Create: `test_location_guardIntersect_polytope_01_basic.py` (GENERATED TEST)
   - Create: `test_location_guardIntersect_conZonotope_01_basic.py` (GENERATED TEST)
   - Test based on MATLAB source code logic:
     - Each method computes intersection correctly
     - Returns correct structure (Rguard, actGuards, minInd, maxInd)
     - Handles different guard set types
     - Handles empty intersections

---

## 10. HYBRID DYNAMICS: location.calcBasis, checkFlow, potInt, potOut

**Status:** Need to check for existing tests

**Files to Test:**
- `cora_python/hybridDynamics/location/calcBasis.py`
- `cora_python/hybridDynamics/location/checkFlow.py`
- `cora_python/hybridDynamics/location/potInt.py`
- `cora_python/hybridDynamics/location/potOut.py`

**Action Plan:**
1. **Search for existing MATLAB tests:**
   - Check if tested indirectly in guardIntersect or reach tests

2. **If tests exist:**
   - Translate existing tests

3. **If no tests exist (GENERATED):**
   - Create: `test_location_calcBasis_01_basic.py` (GENERATED TEST)
   - Create: `test_location_checkFlow_01_basic.py` (GENERATED TEST)
   - Create: `test_location_potInt_01_basic.py` (GENERATED TEST)
   - Create: `test_location_potOut_01_basic.py` (GENERATED TEST)
   - Test based on MATLAB source code logic:
     - calcBasis: Computes orthogonal basis using 'box', 'pca', or 'flow'
     - checkFlow: Removes intersections where flow doesn't point toward guard
     - potInt: Determines which sets potentially intersect guards
     - potOut: Removes parts outside invariant

---

## 11. TEST GENERATION TEMPLATE

For all GENERATED tests, use this template:

```python
"""
GENERATED TEST - No MATLAB test file exists
This test is generated based on MATLAB source code logic.

Source: cora_matlab/[path]/[file].m
Generated: [Date]
"""

import pytest
import numpy as np
from cora_python.[module].[class] import [Class]

def test_[function]_01_basic():
    """
    GENERATED TEST - Basic functionality test
    
    Tests the core functionality of [function] based on MATLAB source code.
    """
    # Setup
    # ... create test objects ...
    
    # Execute
    # ... call function ...
    
    # Verify
    # ... check results match expected behavior ...
    
    # Compare with MATLAB if possible
    # ... if MATLAB output available, compare ...
    
    pass
```

---

## 12. EXECUTION ORDER

1. **Phase 1: Foundation Tests**
   - Test helper functions first (arnoldi, calcBasis, etc.)
   - These are dependencies for other tests

2. **Phase 2: Core Function Tests**
   - Test initReach_adaptive
   - Test taylorMatrices
   - Test outputSet methods

3. **Phase 3: Integration Tests**
   - Test priv_reach_krylov (uses all helpers)
   - Test verify (uses reach methods)
   - Test location.reach (uses guardIntersect methods)
   - Test hybridAutomaton.reach (uses location.reach)

4. **Phase 4: Guard Intersection Tests**
   - Test all guardIntersect methods
   - Test with different guard types
   - Test edge cases

---

## 13. VERIFICATION STRATEGY

For each test:

1. **If MATLAB test exists:**
   - Run MATLAB test and capture output
   - Translate test to Python
   - Compare Python output with MATLAB output
   - Tolerance: `atol=1e-6` for numerical comparisons

2. **If GENERATED test:**
   - Analyze MATLAB source code logic
   - Create test that exercises main code paths
   - Verify output structure matches MATLAB
   - Test edge cases mentioned in MATLAB comments
   - Add comment: `# GENERATED TEST - No MATLAB test file exists`

3. **Integration verification:**
   - Run full workflow tests (e.g., hybridAutomaton.reach)
   - Compare end-to-end results with MATLAB examples
   - Verify all intermediate steps work correctly

---

## 14. TEST FILE NAMING CONVENTION

- **Translated tests:** `test_[ClassName]_[methodName]_[identifier].py`
  - Example: `test_linearSys_verify_01_basic.py`
  
- **Generated tests:** `test_[ClassName]_[methodName]_[identifier].py` (with GENERATED TEST comment)
  - Example: `test_location_guardIntersect_zonoGirard_01_basic.py`

---

## 15. PROGRESS TRACKING

### To Do:
- [ ] Search for existing MATLAB tests for all newly translated code
- [ ] Translate existing MATLAB tests
- [ ] Generate tests for code without MATLAB tests
- [ ] Verify all tests pass
- [ ] Compare results with MATLAB where possible

### Completed:
- [x] Translate: `test_linearSys_reach_krylov_01_heat3D.py` (EXISTING - from benchmark_linear_reach_ARCH23_heat3D_HEAT03.m)
- [x] Generate: `test_linearSys_arnoldi_01_basic.py` (GENERATED TEST)
- [x] Generate: `test_linearSys_priv_initReach_Krylov_01_basic.py` (GENERATED TEST)
- [x] Translate: `test_nonlinearSys_reach_adaptive_01_jetEngine.py` (EXISTING - from example_nonlinear_reach_12_adaptive.m)
- [x] Create: `cora_python/models/Cora/contDynamics/nonlinearSys/models/jetEngine.py` (model file)
- [x] Generate: `test_linearSys_taylorMatrices_01_basic.py` (GENERATED TEST)
- [x] Generate: `test_linearSys_outputSet_01_basic.py` (GENERATED TEST)
- [x] Fix: `cora_python/contDynamics/linearSys/outputSet.py` to return (Y, Verror) tuple
- [x] Fix: `cora_python/contDynamics/contDynamics/outputSet.py` import errors
- [x] Generate: `test_location_calcBasis_01_basic.py` (GENERATED TEST)
- [x] Generate: `test_location_checkFlow_01_basic.py` (GENERATED TEST)
- [x] Generate: `test_location_potInt_01_basic.py` (GENERATED TEST)
- [x] Generate: `test_location_potOut_01_basic.py` (GENERATED TEST)
- [x] Translate: `cora_python/hybridDynamics/transition/transition.py` (missing dependency)
- [x] Translate: `cora_python/hybridDynamics/linearReset/linearReset.py` (missing dependency)
- [x] Create: `cora_python/hybridDynamics/linearReset/evaluate.py` (method in separate file per guidelines)
- [x] Update: `cora_python/hybridDynamics/linearReset/__init__.py` to attach evaluate method
- [x] Update: `cora_python/hybridDynamics/transition/transition.py` to match MATLAB structure (syncLabel, aux_computeProperties)
- [x] Update: `cora_python/hybridDynamics/__init__.py` to export Transition and LinearReset
- [x] Generate: `test_linearReset_01_basic.py` (GENERATED TEST)
- [x] Generate: `test_transition_01_basic.py` (GENERATED TEST)

---

## 16. NOTES

- All GENERATED tests must include the comment: `# GENERATED TEST - No MATLAB test file exists`
- Tests should be based on MATLAB source code logic, not assumptions
- When in doubt, check MATLAB source code comments for expected behavior
- Use pytest fixtures for common test setup
- Follow existing test patterns in `cora_python/tests/`

