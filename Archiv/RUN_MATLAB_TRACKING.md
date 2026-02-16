# Run MATLAB Tracking

## Status

✅ **MATLAB code updated** to track detailed reduction values:
- dHmax
- h_initial
- h_computed
- redIdx
- final_generators
- All intermediate values

## How to Run

### Option 1: From MATLAB Command Window

1. Open MATLAB
2. Navigate to project directory:
   ```matlab
   cd('D:\Bachelorarbeit\Translate_Cora')
   ```
3. Run tracking:
   ```matlab
   track_upstream_matlab
   ```

### Option 2: From Command Line

```bash
matlab -batch "cd('D:\Bachelorarbeit\Translate_Cora'); track_upstream_matlab"
```

### Option 3: Run Script Directly

```matlab
run_matlab_tracking
```

## What It Does

1. Runs reachability analysis with `trackUpstream = true`
2. Captures R before reduction (8 generators expected)
3. Captures detailed reduction algorithm values:
   - dHmax
   - h_initial (initial h values)
   - h_computed (computed h values after processing)
   - redIdx (number of generators to reduce)
   - final_generators (resulting generator count)
4. Saves to `upstream_matlab_log.mat`

## After Running

Run the comparison script:
```bash
python compare_reduction_detailed.py
```

This will show exactly where Python and MATLAB diverge!
