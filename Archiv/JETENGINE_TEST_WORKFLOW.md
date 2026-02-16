# JetEngine Test Workflow

This document describes the workflow to ensure MATLAB and Python run the same way and produce matching results.

## Overview

The test `test_nonlinearSys_reach_adaptive_01_jetEngine.py` verifies that Python produces the same results as MATLAB for the jetEngine model with adaptive reachability analysis.

## Workflow

### Step 1: Generate MATLAB Expected Values

Run MATLAB to generate expected values:

```bash
matlab -batch "generate_jetEngine_expected_values"
```

This creates:
- `jetEngine_expected_values.mat` - MATLAB format with expected values
- `jetEngine_expected_values.txt` - Text format for reference

**Expected values (from MATLAB):**
- `numSteps`: 237
- `finalTime`: 8.0000000000
- `finalRadius`: 5.7960344207e-02
- `options_alg`: 'lin'

### Step 2: Run Python Test

The Python test automatically loads expected values from `jetEngine_expected_values.mat` and verifies:

1. **Input parameters** (must match MATLAB exactly):
   - `params.tFinal = 8.0`
   - `params.R0 = Zonotope(np.array([[1], [1]]), 0.1 * np.eye(2))`
   - `params.U = Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))`
   - `options.alg = 'lin-adaptive'`
   - `options.progress = True`
   - `options.progressInterval = 5`
   - `options.verbose = 1`

2. **Output verification** (with tolerances):
   - Final time: within 0.1 seconds of MATLAB
   - Final radius: within 10% relative tolerance
   - Algorithm: must be 'lin' (after 'adaptive' removal)

### Step 3: Automated Verification

Run the complete workflow:

```bash
python run_jetEngine_verification.py
```

This script:
1. Runs MATLAB to generate expected values
2. Runs Python test
3. Compares results and reports differences

## Files

- `generate_jetEngine_expected_values.m` - MATLAB script to generate expected values
- `test_jetEngine_matlab.m` - MATLAB test script (for manual comparison)
- `test_jetEngine_python.py` - Python test script (for manual comparison)
- `cora_python/tests/contDynamics/nonlinearSys/test_nonlinearSys_reach_adaptive_01_jetEngine.py` - Python pytest test
- `run_jetEngine_verification.py` - Automated workflow script

## Important Notes

1. **Path Requirements**: Both MATLAB scripts add the path to jetEngine hessian functions:
   ```matlab
   addpath('cora_matlab/models/auxiliary/jetEngine');
   ```

2. **Expected Values File**: The Python test requires `jetEngine_expected_values.mat` to exist. If it doesn't exist, the test will skip with a message to run the MATLAB generation script first.

3. **Tolerances**: 
   - Final time: 0.1 seconds absolute tolerance
   - Final radius: 10% relative tolerance
   - These tolerances account for numerical differences between MATLAB and Python implementations

4. **Algorithm Name**: Both MATLAB and Python remove the 'adaptive' suffix from `options.alg` during computation (this is expected behavior).

## Troubleshooting

If the test fails:

1. **Missing expected values file**: Run `generate_jetEngine_expected_values.m` first
2. **Path errors in MATLAB**: Ensure `cora_matlab/models/auxiliary/jetEngine` exists and contains hessian functions
3. **Mismatched results**: Check that input parameters match exactly between MATLAB and Python
