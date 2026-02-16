# Intermediate Value Tracking - Ready for Testing

## Status: ✅ READY

Both Python and MATLAB tracking implementations are complete and ready for testing.

## Quick Start

### 1. Run Python Test

```bash
python test_tracking_jetEngine.py
```

This will:
- Run `reach_adaptive` with the jetEngine model
- Create trace files: `intermediate_values_step{N}_inner_loop.txt`
- Show summary of created files

### 2. Run MATLAB Test

```matlab
test_tracking_jetEngine_matlab
```

Or from command line:
```bash
matlab -batch "test_tracking_jetEngine_matlab"
```

This will:
- Run `reach_adaptive` with the jetEngine model
- Create trace files: `intermediate_values_step{N}_inner_loop.txt`
- Show summary of created files

### 3. Compare Results

```bash
python compare_intermediate_values.py \
    intermediate_values_step1_inner_loop.txt \
    intermediate_values_step1_inner_loop.txt \
    1e-10
```

**Note:** Since both create files with the same name, you may need to:
- Run them in separate directories, OR
- Rename files before comparison (e.g., add `_matlab` or `_python` suffix)

## What's Tracked

Both implementations track the same values:

1. **error_adm**: Input error bound (vector and max)
2. **RallError**: Error solution (center, radius, radius_max)
3. **Rmax**: Combined reachable set (center, radius, radius_max)
4. **Z**: Cartesian product before quadMap (tensorOrder 3 only)
5. **errorSec**: Second-order error after quadMap (tensorOrder 3 only)
6. **VerrorDyn**: Dynamic error set (center, radius, radius_max)
7. **trueError**: Computed error vector (vector and max)
8. **perfIndCurr**: Performance index (value, Inf/NaN status)
9. **perfInds**: Array of performance indices
10. **Convergence status**: Loop convergence/divergence

## File Format

Both implementations create files with identical format:

```
=== Inner Loop Intermediate Values - Step N ===
Initial error_adm_horizon: [values]
Algorithm: lin
TensorOrder: 3

--- Inner Loop Iteration 1 ---
error_adm: [values]
error_adm_max: value
RallError center: [values]
...
```

## Testing the Error Growth Issue

To investigate the `error_adm_horizon` growth issue:

1. **Run with longer time horizon:**
   ```python
   params['tFinal'] = 5.0  # or longer
   ```

2. **Enable tracking:**
   ```python
   options['traceIntermediateValues'] = True
   ```

3. **Compare step by step:**
   - Find the step where `error_adm_horizon` starts growing
   - Compare all intermediate values at that step
   - Identify where MATLAB and Python diverge

4. **Use comparison script:**
   ```bash
   python compare_intermediate_values.py matlab_file.txt python_file.txt 1e-10
   ```

## Implementation Details

- **Python**: See `TRACKING_IMPLEMENTATION_SUMMARY.md`
- **MATLAB**: See `MATLAB_TRACKING_IMPLEMENTATION.md`
- **Comparison Tool**: See `compare_intermediate_values.py`

## Next Steps

1. ✅ **Infrastructure complete** - Both implementations ready
2. ⏳ **Test with problematic case** - Run with longer tFinal to capture error growth
3. ⏳ **Compare step by step** - Use comparison tool to find divergence point
4. ⏳ **Fix discrepancies** - Address any differences found

## Troubleshooting

**No trace files created:**
- Verify `traceIntermediateValues` is enabled
- Check write permissions
- Ensure inner loop is executing

**Files overwrite each other:**
- Run MATLAB and Python in separate directories
- Or rename files after each run

**Values don't match:**
- Check tolerance (may need adjustment)
- Verify same input parameters
- Some numerical differences are expected
