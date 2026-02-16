# Testing Status - Intermediate Value Tracking

## ✅ Infrastructure Complete

All tracking infrastructure is implemented and ready for testing:

### Python Implementation
- ✅ Tracking code in `linReach_adaptive.py`
- ✅ Z/errorSec tracking in `priv_abstractionError_adaptive.py`
- ✅ Test script: `test_tracking_jetEngine.py` (short test)
- ✅ Test script: `test_tracking_jetEngine_long.py` (long test for error growth)
- ✅ Comparison tool: `compare_intermediate_values.py`

### MATLAB Implementation
- ✅ Tracking code in `linReach_adaptive.m`
- ✅ Z/errorSec tracking in `priv_abstractionError_adaptive.m`
- ✅ Test script: `test_tracking_jetEngine_matlab.m`
- ✅ All calls updated to pass `trace_file` parameter

## 🧪 Testing

### Quick Test (Short Time Horizon)
```bash
python test_tracking_jetEngine.py
```
- Runs with `tFinal = 1.0`
- Creates trace files for ~50-60 steps
- Verifies tracking is working

### Long Test (Error Growth Analysis)
```bash
python test_tracking_jetEngine_long.py
```
- Runs with `tFinal = 5.0`
- Designed to capture `error_adm_horizon` growth
- Analyzes growth patterns
- May take several minutes

### MATLAB Test
```matlab
test_tracking_jetEngine_matlab
```
Or:
```bash
matlab -batch "test_tracking_jetEngine_matlab"
```

## 📊 What Gets Tracked

Both implementations track identical values:

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

## 🔍 Comparison Workflow

1. **Run Python test:**
   ```bash
   python test_tracking_jetEngine_long.py
   ```
   Creates: `intermediate_values_step{N}_inner_loop.txt`

2. **Run MATLAB test:**
   ```matlab
   test_tracking_jetEngine_matlab
   ```
   Creates: `intermediate_values_step{N}_inner_loop.txt`

3. **Compare specific step:**
   ```bash
   # Rename files to avoid conflicts
   mv intermediate_values_step10_inner_loop.txt step10_python.txt
   # (MATLAB file already has same name)
   
   python compare_intermediate_values.py step10_python.txt intermediate_values_step10_inner_loop.txt 1e-10
   ```

4. **Analyze divergence:**
   - Find step where `error_adm_horizon` starts growing
   - Compare all intermediate values at that step
   - Identify first value that diverges
   - Trace back to find root cause

## 📝 File Format

Both implementations create identical file format:

```
=== Inner Loop Intermediate Values - Step N ===
Initial error_adm_horizon: [[value1]
 [value2]]
Algorithm: lin
TensorOrder: 3

--- Inner Loop Iteration 1 ---
error_adm: [value1 value2]
error_adm_max: value
RallError center: [value1 value2]
RallError radius: [value1 value2]
RallError radius_max: value
Rmax center: [value1 value2]
Rmax radius: [value1 value2]
Rmax radius_max: value
[TRACKING] Entered tensorOrder 3 path (lin algorithm)
Z (before quadMap) center: [value1 value2 value3]
Z (before quadMap) radius: [value1 value2 value3]
Z (before quadMap) radius_max: value
errorSec (after quadMap) center: [value1 value2]
errorSec (after quadMap) radius: [value1 value2]
errorSec (after quadMap) radius_max: value
VerrorDyn center: [value1 value2]
VerrorDyn radius: [value1 value2]
VerrorDyn radius_max: value
trueError: [value1 value2]
trueError_max: value
perfIndCurr_ratio: [value1 nan]
perfIndCurr: value
perfIndCurr isinf: 0
perfIndCurr isnan: 0
perfIndCurr <= 1: 1
perfIndCounter: 1
perfInds: []
CONVERGED: perfIndCurr <= 1 or ~any(trueError)

=== Inner Loop Complete ===
Lconverged: 1
Total iterations: 1
Final perfIndCurr: value
```

## 🎯 Next Steps

1. ✅ **Infrastructure ready** - All code implemented
2. ⏳ **Run long test** - Capture error growth with `test_tracking_jetEngine_long.py`
3. ⏳ **Run MATLAB test** - Generate MATLAB traces
4. ⏳ **Compare traces** - Use comparison tool to find divergence
5. ⏳ **Fix discrepancies** - Address differences found

## ⚠️ Notes

- **File naming**: Both create files with same name - run in separate directories or rename
- **Performance**: Long tests may take several minutes
- **Memory**: Trace files can be large for long time horizons
- **Tolerance**: May need to adjust comparison tolerance (default 1e-10)

## 📚 Documentation

- `README_INTERMEDIATE_VALUE_TRACKING.md` - Usage guide
- `TRACKING_IMPLEMENTATION_SUMMARY.md` - Python implementation details
- `MATLAB_TRACKING_IMPLEMENTATION.md` - MATLAB implementation details
- `TRACKING_READY_FOR_TESTING.md` - Quick start guide
- `COMPARISON_ERROR_ADM_HORIZON.md` - Error analysis documentation
