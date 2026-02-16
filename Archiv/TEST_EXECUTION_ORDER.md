# Test Execution Order for Verification and Fixing

**Created:** 2025-01-XX  
**Purpose:** Systematic order for running tests to verify and fix issues, starting with low-dependency tests and working upwards.

---

## Overview

This document defines the order in which tests should be executed and fixed, following the dependency hierarchy from the translation plan. Tests are organized by dependency level, starting with foundational components that have no dependencies, and progressing to higher-level components.

---

## Test Execution Phases

### Phase 0: Basic Set Operations (No Dependencies)
**Purpose:** Verify fundamental set operations that everything else depends on.

**Test Categories:**
1. **Interval Operations**
   - `cora_python/tests/contSet/interval/` - All interval tests
   - Basic operations: `plus`, `minus`, `mtimes`, `and_`, `or_`, `contains_`, etc.
   - Set properties: `center`, `dim`, `volume`, `vertices`, etc.

2. **Zonotope Operations**
   - `cora_python/tests/contSet/zonotope/` - All zonotope tests
   - Basic operations: `plus`, `minus`, `mtimes`, `and_`, `or_`, etc.
   - Set operations: `reduce`, `enclose`, `split`, etc.

3. **Polytope Operations**
   - `cora_python/tests/contSet/polytope/` - All polytope tests
   - Basic operations and predicates

4. **Other Basic Sets**
   - `cora_python/tests/contSet/polyZonotope/` - Basic operations
   - `cora_python/tests/contSet/conZonotope/` - Basic operations
   - `cora_python/tests/contSet/conPolyZono/` - Basic operations
   - `cora_python/tests/contSet/ellipsoid/` - Basic operations
   - `cora_python/tests/contSet/emptySet/` - Basic operations
   - `cora_python/tests/contSet/fullspace/` - Basic operations

**Execution Command:**
```powershell
pytest cora_python/tests/contSet/ -v --tb=short > test_output_phase0.txt
```

**Success Criteria:**
- All basic set operation tests pass
- No import errors
- All set operations return correct types

---

### Phase 1: Helper Functions (Minimal Dependencies)
**Purpose:** Verify utility functions used throughout the codebase.

**Test Categories:**
1. **Macros and Utilities**
   - `cora_python/tests/g/functions/helper/dynamics/checkOptions/`
     - `test_getDefaultValue.py`
     - `test_canUseParallelPool.py`
   - `cora_python/tests/g/functions/helper/sets/contSet/contSet/`
     - `test_lin_error2dAB.py`
   - `cora_python/tests/g/functions/helper/sets/contSet/taylm/`
     - `test_initRangeBoundingObjects.py`

2. **Write Functions**
   - `cora_python/tests/g/functions/verbose/write/`
     - `test_writeMatrix.py`
     - `test_matlabFunction.py`

3. **Set Helper Functions**
   - `cora_python/tests/g/functions/helper/sets/contSet/interval/contractors/`
     - `testLong_contract.py` (contract functions)

4. **Tensor Functions**
   - `cora_python/tests/g/functions/helper/dynamics/contDynamics/contDynamics/`
     - `test_generateNthTensor.py`

**Execution Command:**
```powershell
pytest cora_python/tests/g/functions/ -v --tb=short > test_output_phase1.txt
```

**Success Criteria:**
- All helper function tests pass
- Functions can be imported and called correctly

---

### Phase 2: Foundation Functions (Phase 1 from Translation Plan)
**Purpose:** Verify the foundation functions for nonlinear reachability analysis.

**Test Categories:**
1. **Static Error Precomputation**
   - `cora_python/tests/contDynamics/contDynamics/private/`
     - `test_priv_precompStatError.py`

2. **Abstraction Error Functions**
   - `cora_python/tests/contDynamics/contDynamics/private/`
     - `test_priv_abstrerr_lin.py`
     - `test_priv_abstrerr_poly.py`

3. **Tensor Operations**
   - `cora_python/tests/contDynamics/contDynamics/private/`
     - `test_priv_checkTensorRecomputation.py`

**Execution Command:**
```powershell
pytest cora_python/tests/contDynamics/contDynamics/private/ -v --tb=short > test_output_phase2.txt
```

**Success Criteria:**
- All foundation function tests pass
- Error computation matches MATLAB results
- Tensor operations work correctly

---

### Phase 3: Nonlinear System Core (Phase 2 from Translation Plan)
**Purpose:** Verify core nonlinear system reachability functions.

**Test Categories:**
1. **Initial Reachability**
   - `cora_python/tests/contDynamics/nonlinearSys/`
     - `test_nonlinearSys_initReach.py`

2. **Linearization**
   - `cora_python/tests/contDynamics/nonlinearSys/`
     - `test_nonlinearSys_linearize.py`

3. **Tensor Evaluation**
   - `cora_python/tests/contDynamics/nonlinearSys/`
     - `test_nonlinearSys_evalNthTensor.py`
     - `testLong_nonlinearSys_evalNthTensor.py`
     - `testLong_nonlinearSys_tensorCreation.py`

4. **Linear Error**
   - `cora_python/tests/contDynamics/nonlinearSys/`
     - `testLong_nonlinearSys_linError.py`

**Execution Command:**
```powershell
pytest cora_python/tests/contDynamics/nonlinearSys/test_nonlinearSys_initReach.py cora_python/tests/contDynamics/nonlinearSys/test_nonlinearSys_linearize.py cora_python/tests/contDynamics/nonlinearSys/test_nonlinearSys_evalNthTensor.py -v --tb=short > test_output_phase3a.txt
pytest cora_python/tests/contDynamics/nonlinearSys/testLong_nonlinearSys_*.py -v --tb=short > test_output_phase3b.txt
```

**Success Criteria:**
- Initial reachability computation works
- Linearization produces correct results
- Tensor operations match MATLAB

---

### Phase 4: Nonlinear System Reachability (Phase 2 continued)
**Purpose:** Verify full reachability analysis for nonlinear systems.

**Test Categories:**
1. **Basic Reachability**
   - `cora_python/tests/contDynamics/nonlinearSys/`
     - `test_nonlinearSys_reach_01_tank.py`
     - `test_nonlinearSys_reach_02_linearEqualsNonlinear.py`
     - `test_nonlinearSys_reach_04_laubLoomis_polyZonotope.py`
     - `test_nonlinearSys_reach_06_tank_linearRemainder.py`
     - `test_nonlinearSys_reach_time.py`

2. **Approximate Dependent Reachability**
   - `cora_python/tests/contDynamics/nonlinearSys/`
     - `test_nonlinearSys_approxDepReach.py`

3. **Long Reachability Tests**
   - `cora_python/tests/contDynamics/nonlinearSys/`
     - `testLong_nonlinearSys_reach_03_vanDerPol.py`
     - `testLong_nonlinearSys_reach_05_autonomousCar.py`
     - `testLong_nonlinearSys_reach_06_autonomousCar_SRX.py`
     - `testLong_nonlinearSys_reach_07_VDP_linearRemainder.py`
     - `testLong_nonlinearSys_reach_output.py`
     - `testLong_nonlinearSys_reach_poly.py`
     - `testLong_nonlinearSys_reach_time.py`

4. **Inner Approximation**
   - `cora_python/tests/contDynamics/nonlinearSys/`
     - `testLong_nonlinearSys_reachInner.py`
     - `testLong_nonlinearSys_reachInner_02_minkdiff.py`

**Execution Command:**
```powershell
pytest cora_python/tests/contDynamics/nonlinearSys/test_nonlinearSys_reach_*.py -v --tb=short > test_output_phase4a.txt
pytest cora_python/tests/contDynamics/nonlinearSys/testLong_nonlinearSys_reach*.py -v --tb=short > test_output_phase4b.txt
```

**Success Criteria:**
- Reachability analysis produces correct results
- Results match MATLAB output
- All algorithms (standard, poly, linRem) work

---

### Phase 5: Linear System Verification (Phase 3 from Translation Plan)
**Purpose:** Verify linear system verification algorithms.

**Test Categories:**
1. **Basic Linear System Tests**
   - `cora_python/tests/contDynamics/linearSys/`
     - `test_linearSys_reach.py`
     - `test_linearSys_reach_adaptive.py`
     - `test_linearSys_affineSolution.py`
     - `test_linearSys_simulate.py`
     - `test_linearSys_verify_01_temporalLogic.py` (may skip if STL not implemented)

2. **Fast Verification Benchmarks**
   - `cora_python/tests/contDynamics/linearSys/`
     - `test_verifyFast_beam_CBC01.py`
     - `test_verifyFast_beam_CBC02.py`
     - `test_verifyFast_beam_CBC03.py`
     - `test_verifyFast_beam_CBF01.py`
     - `test_verifyFast_beam_CBF02.py`
     - `test_verifyFast_beam_CBF03.py`
     - `test_verifyFast_heat3D_HEAT01.py`
     - `test_verifyFast_heat3D_HEAT02.py`
     - `test_verifyFast_iss_ISSC01_ISS02.py`
     - `test_verifyFast_iss_ISSC01_ISU02.py`
     - `test_verifyFast_iss_ISSF01_ISS01.py`
     - `test_verifyFast_iss_ISSF01_ISU01.py`

**Execution Command:**
```powershell
pytest cora_python/tests/contDynamics/linearSys/test_linearSys_*.py -v --tb=short > test_output_phase5a.txt
pytest cora_python/tests/contDynamics/linearSys/test_verifyFast_*.py -v --tb=short > test_output_phase5b.txt
```

**Success Criteria:**
- Linear system reachability works
- Verification algorithms produce correct results
- Benchmarks match MATLAB results

---

### Phase 6: Hybrid Dynamics (Phase 4 from Translation Plan)
**Purpose:** Verify hybrid system reachability analysis.

**Test Categories:**
1. **Nonlinear Reset**
   - `cora_python/tests/hybridDynamics/nonlinearReset/`
     - `test_nonlinearReset_derivatives.py`

2. **Hybrid Automaton** (when implemented)
   - `cora_python/tests/hybridDynamics/hybridAutomaton/` (future)

**Execution Command:**
```powershell
pytest cora_python/tests/hybridDynamics/ -v --tb=short > test_output_phase6.txt
```

**Success Criteria:**
- Hybrid system operations work
- Reset functions compute correctly

---

### Phase 7: ContDynamics Base Class
**Purpose:** Verify base class functionality.

**Test Categories:**
1. **ContDynamics Base**
   - `cora_python/tests/contDynamics/contDynamics/`
     - `test_contDynamics.py`
     - `test_contDynamics_display.py`
     - `test_contDynamics_simulateRandom.py`
     - `test_symVariables.py`

2. **Private Functions**
   - `cora_python/tests/contDynamics/contDynamics/private/`
     - `test_deleteRedundantSets.py`
     - `test_priv_select.py`

**Execution Command:**
```powershell
pytest cora_python/tests/contDynamics/contDynamics/ -v --tb=short > test_output_phase7.txt
```

**Success Criteria:**
- Base class methods work correctly
- Inheritance hierarchy functions properly

---

### Phase 8: Other Systems
**Purpose:** Verify other system types.

**Test Categories:**
1. **Linear ARX**
   - `cora_python/tests/contDynamics/linearARX/`

2. **Nonlinear ARX**
   - `cora_python/tests/contDynamics/nonlinearARX/`

3. **Nonlinear Sys DT**
   - `cora_python/tests/contDynamics/nonlinearSysDT/`

**Execution Command:**
```powershell
pytest cora_python/tests/contDynamics/linearARX/ cora_python/tests/contDynamics/nonlinearARX/ cora_python/tests/contDynamics/nonlinearSysDT/ -v --tb=short > test_output_phase8.txt
```

**Success Criteria:**
- All system types work correctly

---

### Phase 9: Specification and Other
**Purpose:** Verify specification handling and other components.

**Test Categories:**
1. **Specification**
   - `cora_python/tests/specification/`
     - `test_syntaxTree.py`

2. **Other Components**
   - `cora_python/tests/converter/`
   - `cora_python/tests/matrixSet/`

**Execution Command:**
```powershell
pytest cora_python/tests/specification/ cora_python/tests/converter/ cora_python/tests/matrixSet/ -v --tb=short > test_output_phase9.txt
```

**Success Criteria:**
- Specification handling works
- All components function correctly

---

## Execution Strategy

### Step-by-Step Process

1. **Start with Phase 0**
   - Run all Phase 0 tests
   - Fix any failures before proceeding
   - Document any issues

2. **Progress Sequentially**
   - Complete each phase before moving to the next
   - Fix all failures in a phase before proceeding
   - Update TODO list as tests pass

3. **Document Failures**
   - For each failing test, create a debug script
   - Compare Python vs MATLAB results
   - Fix issues systematically

4. **Integration Testing**
   - After all phases pass, run full test suite
   - Verify no regressions
   - Check performance

### Quick Test Commands

**Run all tests in a phase:**
```powershell
pytest cora_python/tests/[phase_directory]/ -v --tb=short > test_output_phase[X].txt
```

**Run specific test file:**
```powershell
pytest cora_python/tests/[path]/test_[name].py -v --tb=short
```

**Run only failed tests:**
```powershell
pytest --lf -v
```

**Run with minimal output:**
```powershell
pytest -q --tb=no
```

---

## Notes

- **Dependency Order:** Tests are ordered by dependency level, ensuring foundational components are verified before dependent components.
- **Translation Plan Alignment:** This order matches the translation plan phases, ensuring tests are verified in the same order as code translation.
- **Future Tests:** When new code is translated, its tests should be added to the appropriate phase based on dependencies.
- **Continuous Integration:** This order can be used for CI/CD pipelines to ensure tests pass in the correct sequence.

---

## Maintenance

- **Update when:** New test phases are added, dependency structure changes, or translation plan is updated.
- **Review frequency:** After each major translation milestone.

