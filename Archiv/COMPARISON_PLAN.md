# MATLAB-Python Comparison Plan

## Test Setup

### MATLAB Test (`test_matlab_comparison_working.m`)
- **Model**: `tank6Eq` (6 states, 1 input)
- **Time horizon**: 0.0 to 2.0
- **Tensor order**: 2
- **Initial set**: `zonotope([2; 4; 4; 2; 10; 4], 0.2*eye(6))`
- **Input set**: `zonotope([0,0.005])`
- **Status**: ✅ Completed - Generated 897 trace files
- **Growth**: 17.7 trillion times (matches Python pattern)

### Python Test (`test_python_comparison_tank6Eq.py`)
- **Model**: `tank6Eq` (6 states, 1 input) - same dynamics
- **Time horizon**: 0.0 to 2.0
- **Tensor order**: 2
- **Initial set**: Same as MATLAB
- **Input set**: Same as MATLAB
- **Status**: ⏳ Running

## Comparison Process

1. **Run Python test** - Generate trace files with same parameters
2. **Extract values** from both MATLAB and Python trace files:
   - `initial_error_adm_horizon`
   - `run1_trueError_max`
   - `run1_error_adm_horizon_set`
   - `run2_trueError_max`
   - `run2_error_adm_Deltatopt`
   - `final_error_adm_horizon`
3. **Compare step by step** - Identify differences
4. **Trace differences** - Find where values diverge

## Comparison Script

`compare_matlab_python_traces.py` will:
- Find common steps between MATLAB and Python
- Extract all tracked values from each trace file
- Compare values with tolerance (default 1e-6)
- Report absolute and relative differences
- Identify steps where values diverge

## Expected Results

Based on code analysis:
- **Code logic matches**: Both set `error_adm_horizon` from Run 1's `trueError`
- **Growth pattern should match**: Both should show exponential growth
- **Values should match**: Intermediate values should be identical or very close

## Next Steps

1. Wait for Python test to complete
2. Run comparison script: `python compare_matlab_python_traces.py`
3. Analyze any differences found
4. Document findings
