# TODO: Verify and Fix Generated Tests

**Created:** 2025-01-XX  
**Purpose:** All tests marked as "GENERATED" need MATLAB verification to ensure they correctly test the MATLAB logic and use correct expected values.

---

## Overview

Generated tests are tests that were created without a corresponding MATLAB test file. These tests need:
1. MATLAB scripts to verify the test logic is correct
2. MATLAB execution to produce correct input/output pairs
3. Verification that the test expectations match MATLAB behavior
4. Integration of MATLAB-generated I/O pairs into Python tests

---

## Generated Test Files (58 files found)

### Priority 1: Critical Functionality Tests

#### Hybrid Dynamics
- [ ] `cora_python/tests/hybridDynamics/linearReset/test_linearReset_linearReset.py`
- [ ] `cora_python/tests/hybridDynamics/linearReset/test_linearReset_01_basic.py`
- [ ] `cora_python/tests/hybridDynamics/hybridAutomaton/test_hybridAutomaton_priv_isFinalLocation.py`
- [ ] `cora_python/tests/hybridDynamics/hybridAutomaton/test_hybridAutomaton_derivatives.py`
- [ ] `cora_python/tests/hybridDynamics/location/test_location_derivatives.py`
- [ ] `cora_python/tests/hybridDynamics/location/test_location_eventFcn.py`
- [ ] `cora_python/tests/hybridDynamics/location/test_location_guardIntersect*.py` (6 files)
- [ ] `cora_python/tests/hybridDynamics/location/test_location_checkFlow_helpers.py`
- [ ] `cora_python/tests/hybridDynamics/location/test_location_calcBasis_helpers.py`
- [ ] `cora_python/tests/hybridDynamics/transition/test_transition_convGuard.py`
- [ ] `cora_python/tests/hybridDynamics/transition/test_transition_guard2polytope.py`
- [ ] `cora_python/tests/hybridDynamics/transition/test_transition_eventFcn.py`
- [ ] `cora_python/tests/hybridDynamics/nonlinearReset/test_nonlinearReset_derivatives.py`

#### Linear System
- [x] `cora_python/tests/contDynamics/linearSys/private/test_priv_initReach_Krylov_01_basic.py` ✅ **VERIFIED** - MATLAB I/O pairs from debug_matlab_initReach_Krylov.m (accessed via private directory)
- [x] `cora_python/tests/contDynamics/linearSys/test_linearSys_arnoldi_01_basic.py` ✅ **VERIFIED** - MATLAB I/O pairs from debug_matlab_arnoldi.m
- [x] `cora_python/tests/contDynamics/linearSys/test_linearSys_taylorMatrices_01_basic.py` ✅ **VERIFIED** - MATLAB I/O pairs from debug_matlab_taylorMatrices.m
- [x] `cora_python/tests/contDynamics/linearSys/test_linearSys_outputSet_01_basic.py` ✅ **VERIFIED** - MATLAB I/O pairs from debug_matlab_outputSet.m

#### Continuous Dynamics Foundation
- [ ] `cora_python/tests/contDynamics/contDynamics/test_symVariables.py`
- [ ] `cora_python/tests/contDynamics/contDynamics/private/test_priv_select.py`
- [ ] `cora_python/tests/contDynamics/contDynamics/private/test_priv_precompStatError.py`
- [ ] `cora_python/tests/contDynamics/contDynamics/private/test_priv_checkTensorRecomputation.py`
- [ ] `cora_python/tests/contDynamics/contDynamics/private/test_priv_abstrerr_lin.py`
- [ ] `cora_python/tests/contDynamics/contDynamics/private/test_priv_abstrerr_poly.py`
- [ ] `cora_python/tests/contDynamics/contDynamics/private/test_deleteRedundantSets.py`

#### Nonlinear System
- [ ] `cora_python/tests/contDynamics/nonlinearSys/testLong_nonlinearSys_reach_06_autonomousCar_SRX.py`

### Priority 2: Helper Functions

- [ ] `cora_python/tests/g/functions/helper/dynamics/checkOptions/test_getDefaultValue.py`
- [ ] `cora_python/tests/g/functions/helper/dynamics/checkOptions/test_canUseParallelPool.py`
- [ ] `cora_python/tests/g/functions/helper/dynamics/contDynamics/contDynamics/test_generateNthTensor.py`
- [ ] `cora_python/tests/g/functions/helper/sets/contSet/contSet/test_lin_error2dAB.py`
- [ ] `cora_python/tests/g/functions/helper/sets/contSet/taylm/test_initRangeBoundingObjects.py`
- [ ] `cora_python/tests/g/functions/matlab/function_handle/test_isequalFunctionHandle.py`
- [ ] `cora_python/tests/g/functions/matlab/indexing/test_batchCombinator.py`
- [ ] `cora_python/tests/g/functions/verbose/write/test_writeMatrix.py`
- [ ] `cora_python/tests/g/functions/verbose/write/test_matlabFunction.py`

### Priority 3: Set Operations

- [ ] `cora_python/tests/contSet/contSet/test_generateRandom.py`
- [ ] `cora_python/tests/contSet/ellipsoid/test_ellipsoid_mtimes.py`
- [ ] `cora_python/tests/contSet/ellipsoid/test_ellipsoid_or.py`
- [ ] `cora_python/tests/contSet/interval/test_interval_spectraShadow.py`
- [ ] `cora_python/tests/contSet/polyZonotope/test_polyZonotope_approxVolumeRatio.py`
- [ ] `cora_python/tests/contSet/polyZonotope/test_polyZonotope_generateRandom.py`
- [ ] `cora_python/tests/contSet/polytope/testLong_polytope_contains_.py`
- [ ] `cora_python/tests/contSet/probZonotope/test_probZonotope_generateRandom.py`
- [ ] `cora_python/tests/contSet/spectraShadow/test_spectraShadow_generateRandom.py`
- [ ] `cora_python/tests/contSet/taylm/test_taylm_isemptyobject.py`
- [ ] `cora_python/tests/contSet/zonoBundle/test_zonoBundle_generateRandom.py`
- [ ] `cora_python/tests/contSet/zonoBundle/test_zonoBundle_isemptyobject.py`

### Priority 4: Neural Networks

- [ ] `cora_python/tests/nn/layers/nonlinear/test_nnActivationLayer.py`
- [ ] `cora_python/tests/nn/layers/nonlinear/test_nnSigmoidLayer.py`
- [ ] `cora_python/tests/nn/layers/nonlinear/test_nnSoftmaxLayer.py`
- [ ] `cora_python/tests/nn/layers/nonlinear/test_nnTanhLayer.py`

### Priority 5: Other

- [ ] `cora_python/tests/g/classes/test_testCase_sequential.py`
- [ ] `cora_python/tests/specification/test_syntaxTree.py`

---

## Workflow for Each Generated Test

### Step 1: Create MATLAB Verification Script
Create `debug_matlab_[test_name].m` that:
1. Runs the same test logic as the Python test
2. Outputs all intermediate values
3. Saves exact input/output pairs to a text file
4. Verifies the test logic is correct

### Step 2: Execute MATLAB Script
```powershell
matlab -batch "run('debug_matlab_[test_name].m')"
```

### Step 3: Extract I/O Pairs
1. Read the MATLAB output file
2. Extract exact numeric values
3. Note data types (double, zonotope, interval, etc.)
4. Document tolerance used in MATLAB

### Step 4: Update Python Test
1. Replace generated test values with MATLAB I/O pairs
2. Use exact MATLAB tolerance
3. Add comment indicating source: `# MATLAB I/O pairs from debug_matlab_[test_name].m`
4. Ensure test logic matches MATLAB test (if MATLAB test exists)

### Step 5: Verify Test Passes
1. Run Python test
2. Compare results against MATLAB output
3. If differences exist, investigate root cause:
   - Check numerical precision
   - Verify algorithm implementation matches MATLAB
   - Check data type conversions
   - Verify tolerance is appropriate

---

## Example: affineSolution Test

**Status:** ⚠️ Needs tolerance adjustment

**MATLAB Results:**
- `Pu` type: `double` (numeric array)
- `Pu_true` = `[0.101720693618414; -0.0388092851102899]`
- `Pu` (computed) = `[0.101720693618414; -0.0388092851102899]`
- Max difference: `0` (exact match in MATLAB)
- Tolerance used: `1e-14`

**Python Results:**
- `Pu` type: `Zonotope` (Python returns zonotope, MATLAB returns numeric)
- Difference: `~4e-11` (numerical precision issue)
- Issue: Python implementation returns zonotope when MATLAB returns numeric, OR tolerance needs adjustment

**Action Required:**
1. Check if Python `affineSolution` should return numeric instead of zonotope
2. OR adjust tolerance to account for numerical differences
3. Verify against MATLAB source code

---

## Notes

- **Always compare against MATLAB**: Never trust generated tests without MATLAB verification
- **Root cause analysis**: When tests fail, compare step-by-step against MATLAB, not just the final result
- **Debug scripts are essential**: Create both MATLAB and Python debug scripts to trace execution
- **I/O pairs must be exact**: Use exact MATLAB values, not approximations
- **Tolerance matters**: Use the same tolerance as MATLAB, or document why it differs

---

## Progress Tracking

- **Total Generated Tests:** 58
- **Verified with MATLAB:** 5 (affineSolution ✅, taylorMatrices ✅, arnoldi ✅, outputSet ✅, initReach_Krylov ✅)
- **Fully Fixed:** 5
- **Remaining:** 53
