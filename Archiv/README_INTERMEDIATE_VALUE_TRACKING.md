# Intermediate Value Tracking for Error Analysis

This document explains how to use the intermediate value tracking feature to compare MATLAB and Python implementations step by step.

## Overview

The tracking feature logs all key intermediate values in the inner loop of `linReach_adaptive` to help identify where MATLAB and Python diverge when `error_adm_horizon` grows unbounded.

## Enabling Tracking

### Python

Add to your options dictionary:
```python
options['traceIntermediateValues'] = True
```

When enabled, files will be created: `intermediate_values_step{N}_inner_loop.txt` for each step.

### MATLAB

Add to your options struct:
```matlab
options.traceIntermediateValues = true;
```

When enabled, files will be created: `intermediate_values_step{N}_inner_loop.txt` for each step.

**Note:** The MATLAB tracking code has been integrated into `linReach_adaptive.m` and `priv_abstractionError_adaptive.m`. See `MATLAB_TRACKING_IMPLEMENTATION.md` for details.

## Tracked Values

At each inner loop iteration, the following values are logged:

1. **error_adm**: Input error bound (vector and max)
2. **RallError**: Error solution from `errorSolution_adaptive` (center, radius, radius_max)
3. **Rmax**: Combined reachable set `Rlinti + RallError` (center, radius, radius_max)
4. **Z**: Cartesian product `Rred.cartProd_(U)` before `quadMap` (center, radius, radius_max)
5. **errorSec**: Second-order error after `quadMap` (center, radius, radius_max)
6. **VerrorDyn**: Dynamic error set after reduction (center, radius, radius_max)
7. **trueError**: Computed error vector (vector and max)
8. **perfIndCurr**: Performance index `max(trueError ./ error_adm)` (value, Inf/NaN status, comparison)
9. **perfInds**: Array of performance indices for divergence detection
10. **Convergence status**: Whether loop converged or diverged

## Comparing Results

Use the comparison script to compare MATLAB and Python traces:

```bash
python compare_intermediate_values.py matlab_trace.txt python_trace.txt 1e-10
```

The script will:
- Parse both trace files
- Compare all tracked values
- Report matches, mismatches, and missing values
- Use configurable tolerance for numerical comparisons

## Example Usage

### Python Test

```python
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.models.Cora.contDynamics.nonlinearSys.models.jetEngine import jetEngine

sys = NonlinearSys(name='jetEngine', fun=jetEngine, states=2, inputs=1)

params = {
    'tFinal': 1.0,
    'R0': Zonotope(np.array([[1], [1]]), 0.1 * np.eye(2)),
    'U': Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))
}

options = {
    'alg': 'lin-adaptive',
    'traceIntermediateValues': True,  # Enable tracking
    'progress': True
}

R, _, opt = sys.reach(params, options)
```

### Running Comparison

```bash
# Run Python test
python test_tracking_jetEngine.py

# Run MATLAB equivalent (with tracking enabled)
matlab -batch "test_tracking_jetEngine_matlab"

# Compare results
python compare_intermediate_values.py \
    intermediate_values_step1_inner_loop.txt \
    intermediate_values_step1_inner_loop.txt \
    1e-10
```

**Note:** Both MATLAB and Python create files with the same name format. You may want to rename them or run them in separate directories for comparison.

## File Format

Trace files contain:
- Header with step number
- Section for each inner loop iteration: `--- Inner Loop Iteration N ---`
- Key-value pairs for each tracked value
- Summary at end with convergence status

Example:
```
=== Inner Loop Intermediate Values - Step 1 ===
Initial error_adm_horizon: [1e-10, 1e-10]

--- Inner Loop Iteration 1 ---
error_adm: [1.000000e-10 1.000000e-10]
error_adm_max: 1.000000e-10
RallError center: [0.000000e+00 0.000000e+00]
RallError radius: [1.234567e-10 1.234567e-10]
RallError radius_max: 1.234567e-10
...
```

## Troubleshooting

**No trace files created:**
- Verify `options['traceIntermediateValues'] = True` is set
- Check that the inner loop is actually executing
- Ensure write permissions in the current directory

**Values don't match:**
- Check tolerance used in comparison (may need to adjust)
- Verify both MATLAB and Python are using same input parameters
- Check for numerical precision differences (expected for floating point)

**Missing values:**
- Some values may not be computed in certain code paths
- Check if the iteration path matches between MATLAB and Python

## Next Steps

1. Run tracking on the jetEngine case that shows the error_adm_horizon growth issue
2. Compare intermediate values step by step to find where divergence occurs
3. Fix any discrepancies found in the comparison
4. Verify the fix resolves the unbounded growth issue
