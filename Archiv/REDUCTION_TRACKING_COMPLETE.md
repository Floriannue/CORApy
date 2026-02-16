# Reduction Tracking - Complete

## Status: ✅ Python Tracking Complete

### Captured Data (Step 3)

**Input:**
- R before reduction: 8 generators
- redFactor: 0.0005
- diagpercent: 0.02236... (sqrt(0.0005))

**Intermediate Values:**
- dHmax: 6.949e-03
- Gbox_sum: 0.2197
- h_initial: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.250e-08, 2.903e-06]
  - 6 generators with h=0
  - 2 generators with h > 0
- hzeroIdx: [0, 1, 2, 7, 5, 6]
- last0Idx: 6
- gensred_shape: (2, 2) - 2 generators remaining after removing zeros
- h_computed: [7.952e-06, 8.299e-06]
  - Both values <= dHmax (6.949e-03)
- redIdx_arr: [0, 1]
- redIdx_0based: 1
- redIdx: 2 (1-based, meaning reduce both generators)
- dHerror: 8.299e-06

**Output:**
- Rred after reduction: 2 generators
- final_generators: 2
- gredIdx: [0, 1, 2, 7, 5, 6, 4, 3] (all 8 generators reduced)

## Key Findings

1. **All generators are reduced**: The algorithm reduces all 8 generators because:
   - 6 generators have h=0 (automatically reduced)
   - 2 generators have h > 0 but both satisfy h <= dHmax

2. **The algorithm is deterministic**: Given the same inputs, it produces the same outputs

3. **Python reduces to 2 generators**: This matches our earlier observation

## Next Steps

1. **Re-run MATLAB tracking** to capture:
   - R before reduction (should be 8 generators)
   - dHmax value
   - h_initial values
   - h_computed values
   - redIdx value
   - Rred after reduction (currently 4 generators)

2. **Compare values**:
   - If dHmax differs → check redFactor and Gbox calculation
   - If h_initial differs → check sorting/indexing (mink behavior)
   - If h_computed differs → check floating point precision/BLAS differences
   - If redIdx differs → check threshold logic

3. **Identify divergence point**:
   - The comparison will show exactly where Python and MATLAB diverge
   - This will pinpoint the bug causing the 2 vs 4 generator difference

## Files

- `check_detailed_reduction.py` - Script to view detailed reduction data
- `compare_reduction_inputs.py` - Comparison script (needs MATLAB data)
- `REDUCTION_ANALYSIS.md` - Complete analysis document
