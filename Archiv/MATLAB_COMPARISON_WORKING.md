# MATLAB Comparison - Working Setup

## Fixes Applied

### 1. Model Selection
- **Changed from**: `jetEngine`, `vanDerPol` (missing hessian functions)
- **Changed to**: `tank6Eq` (has `hessianTensorInt_tank6Eq.m` and `thirdOrderTensorInt_tank6Eq.m`)

### 2. Path Configuration
```matlab
addpath(genpath('cora_matlab/models/auxiliary'));
```

### 3. System Creation
```matlab
% Use explicit name to ensure function lookup works
sys = nonlinearSys('tank6Eq',@tank6Eq,6,1);
```

### 4. Required Options Added
```matlab
options.redFactor = 0.5;  % Required for reduce('adaptive')
options.zetaK = 1.5;      % Threshold for tensor order selection
```

### 5. Required Parameters Added
```matlab
params.uTrans = center(params.U);  % Required for linearization
```

## Test Script

The working test script is: `test_matlab_comparison_working.m`

## Status

✅ **MATLAB test is now running** - All initialization errors resolved

## Next Steps

1. Wait for MATLAB test to complete
2. Compare trace files with Python output
3. Analyze `error_adm_horizon` growth patterns
4. Verify both implementations show the same behavior

## Key Learnings

1. **Model requirements**: Models need complete hessian/tensor function sets in `models/auxiliary/<modelName>/`
2. **Path setup**: Must add `models/auxiliary` to MATLAB path
3. **System naming**: Use explicit name in constructor for reliable function lookup
4. **Options**: Many options have defaults but some must be set explicitly
5. **Parameters**: `uTrans` is required even if not used in dynamics
