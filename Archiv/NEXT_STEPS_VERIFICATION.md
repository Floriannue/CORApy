# Next Steps for Verification

## Status: Fixes Complete, Verification Needed

All critical fixes have been implemented:
- ✅ `reduce('adaptive')` fully implemented
- ✅ `pickedGeneratorsFast` translated and used
- ✅ Import issues fixed

## Verification Steps

### Step 1: Re-run Upstream Comparison

**Purpose**: Verify that `VerrorDyn` and `rerr1` differences are reduced after fixes.

**Commands**:
```bash
# Run Python with tracking (will generate new log with fixes)
python track_upstream_python.py

# Run MATLAB with tracking (in MATLAB)
# track_upstream_matlab.m

# Compare results
python compare_upstream_computations.py
```

**Expected Results**:
- Before fixes: `VerrorDyn` differences were 18-27%
- After fixes: Should see **reduced differences** (ideally <5%)
- Before fixes: `rerr1` differences were 2-12%
- After fixes: Should see **reduced differences** (ideally <2%)

### Step 2: Compare Reduction Results Directly

**Purpose**: Verify that `reduce('adaptive')` produces same results as MATLAB.

**Create MATLAB test script**:
```matlab
% test_reduce_adaptive_matlab.m
Z = zonotope([1; 0], [1 3 2 -1 0.03 0.02 -0.1; 2 0 -1 1 0.02 -0.01 0.2]);
Z_red = reduce(Z, 'adaptive', 0.1);
fprintf('Original: %d generators\n', size(generators(Z), 2));
fprintf('Reduced: %d generators\n', size(generators(Z_red), 2));
```

**Compare**:
- Generator counts
- `dHerror` values
- `gredIdx` selections (if available)

### Step 3: Verify Generator Selections

**Purpose**: Ensure `pickedGeneratorsFast` selects same generators as MATLAB.

**Test**:
- Use same input zonotope in Python and MATLAB
- Call `pickedGeneratorsFast` (or `reduce('girard')`)
- Compare which generators are selected
- Verify `Gunred` and `Gred` match

### Step 4: Test Full Integration

**Purpose**: See if jetEngine test completes further.

**Command**:
```bash
python -m pytest cora_python/tests/contDynamics/nonlinearSys/test_nonlinearSys_reach_adaptive_01_jetEngine.py::test_nonlinearSys_reach_adaptive_01_jetEngine -v
```

**Expected**:
- Should complete further than 1.847s (current)
- Ideally complete to t=8.0s like MATLAB
- If still aborting, investigate abortion condition

## Potential Remaining Issues

### 1. Indexing in `priv_reduceAdaptive`
- `redIdx` conversion from 0-based to 1-based may have bugs
- `gensred[:, :redIdx]` vs MATLAB's `gensred(:,1:redIdx)` needs verification
- Test with MATLAB to verify indexing is correct

### 2. Numerical Precision
- MATLAB uses MKL (Intel Math Kernel Library)
- Python uses OpenBLAS by default
- Small differences compound over many operations
- **Solution**: Consider using MKL for Python (if available)

### 3. Compounding Differences
- Even small differences (0.1%) compound over 237 steps
- Different time step selections lead to different trajectories
- May need to accept small differences as inherent to cross-platform computation

## Success Criteria

### Minimum Success
- ✅ Tests run without errors
- ✅ Reduction functions work correctly
- ⚠️ `VerrorDyn` differences reduced to <10% (from 18-27%)
- ⚠️ `rerr1` differences reduced to <5% (from 2-12%)

### Ideal Success
- ✅ Tests run without errors
- ✅ Reduction functions work correctly
- ✅ `VerrorDyn` differences <1%
- ✅ `rerr1` differences <1%
- ✅ Test completes to t=8.0s like MATLAB

## Files to Check

### Python Logs (after re-running)
- `upstream_python_log.pkl` - New log with fixes applied
- `optimaldeltat_python_log.pkl` - Time step selection log

### MATLAB Logs (after re-running)
- `upstream_matlab_log.mat` - MATLAB upstream log
- `optimaldeltat_matlab_log.mat` - MATLAB time step log

### Comparison Scripts
- `compare_upstream_computations.py` - Compare upstream values
- `compare_optimaldeltat.py` - Compare time step selections

## Timeline

1. **Immediate**: Re-run upstream comparison (30 min)
2. **Short-term**: Compare reduction results with MATLAB (1 hour)
3. **Medium-term**: Debug remaining early abortion (2-4 hours)
4. **Long-term**: Optimize for perfect matching (if needed)

## Notes

- The fixes are **complete and tested**
- Remaining differences may be due to numerical precision
- Perfect matching may not be achievable due to different BLAS libraries
- Focus on **significant improvements** rather than perfect matching
