# Reduction Algorithm Analysis

## Python Step 3 Reduction (Detailed Trace)

### Input
- **R before reduction**: 8 generators
- **redFactor**: 0.0005
- **diagpercent**: 0.02236... (sqrt(0.0005))

### Algorithm Execution

1. **Compute dHmax**:
   - Gbox sum: 0.2197
   - dHmax = 6.949e-03

2. **Girard method - Initial sorting**:
   - norminf: max = 1.111e-01
   - normsum: max = 1.111e-01
   - diff = normsum - norminf: max = 2.903e-06
   - idx (sorted): [0, 1, 2, 7, 5, 6, 4, 3]
   - h (initial): [0, 0, 0, 0, 0, 0, 3.250e-08, 2.903e-06]
   - All h <= dHmax: True

3. **Process generators**:
   - hzeroIdx: [0, 1, 2, 7, 5, 6] (6 generators with h=0)
   - last0Idx: 6
   - gensred: 2 generators (remaining after removing 6 zeros)
   - maxidx: [1, 0]
   - maxval: [3.538e-08, 4.011e-06]

4. **Compute new h values**:
   - gensdiag shape: (2, 2)
   - h: [7.952e-06, 8.299e-06]
   - Both h <= dHmax: True
   - redIdx_arr: [0, 1]
   - redIdx (1-based): 2

5. **Final generators**:
   - Gred: (2, 1) - sum of 2 reduced generators
   - Gunred: (2, 0) - no unreduced generators
   - G_diag: (2, 2) - diagonal matrix from Gred + Gzeros
   - **Final: 2 generators**

### Result
- **Rred after reduction**: 2 generators
- **dHerror**: 8.299e-06
- **gredIdx**: [0, 1, 2, 7, 5, 6, 4, 3] (all 8 generators)

## Key Observations

1. **All generators are reduced**: The algorithm reduces all 8 generators (6 with h=0, 2 with h > 0 but <= dHmax)

2. **dHmax is very small**: 6.949e-03, which means the algorithm is very conservative

3. **The algorithm is deterministic**: Given the same inputs, it should produce the same outputs

## Why MATLAB Might Produce 4 Generators

Possible causes for the difference:

1. **Different dHmax calculation**: 
   - Check if redFactor is the same
   - Check if Gbox calculation matches

2. **Different h values**:
   - Floating point precision differences
   - Different BLAS implementations (MKL vs OpenBLAS)

3. **Different sorting behavior**:
   - MATLAB's `mink` vs Python's `argpartition` + `argsort`
   - Handling of ties in sorting

4. **Different threshold check**:
   - MATLAB's `find(h <= dHmax, 1, 'last')` might behave differently
   - Edge case handling when h values are very close to dHmax

## Next Steps

1. **Re-run MATLAB tracking** to capture:
   - R before reduction (should be 8 generators)
   - dHmax value
   - h values at each step
   - redIdx value
   - Rred after reduction (currently 4 generators)

2. **Compare intermediate values**:
   - dHmax: Python vs MATLAB
   - Initial h values: Python vs MATLAB
   - Computed h values: Python vs MATLAB
   - redIdx: Python vs MATLAB

3. **Identify the divergence point**:
   - If dHmax differs → check redFactor and Gbox calculation
   - If h values differ → check floating point precision
   - If redIdx differs → check threshold logic

## Files

- `debug_reduction_detailed.py` - Detailed trace of Python reduction
- `compare_reduction_inputs.py` - Comparison script (needs MATLAB data)
- `REDUCTION_TRACKING_STATUS.md` - Status of tracking implementation
