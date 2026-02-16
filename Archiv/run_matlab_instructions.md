# Instructions to Run MATLAB Tracking

## Manual Execution Required

Due to MATLAB startup issues, please run the tracking script manually:

### Option 1: MATLAB Command Window

1. Open MATLAB
2. Navigate to project directory:
   ```matlab
   cd('D:\Bachelorarbeit\Translate_Cora')
   ```
3. Run tracking:
   ```matlab
   track_upstream_matlab
   ```

### Option 2: MATLAB Script File

1. Open MATLAB
2. Open the file `track_upstream_matlab.m`
3. Run it (F5 or click Run)

## What It Does

The script will:
1. Run reachability analysis with `trackUpstream = true`
2. Capture detailed tracking data including `initReach_tracking` for all steps/runs
3. Save to `upstream_matlab_log.mat`

## After Running

Once `upstream_matlab_log.mat` is updated, run:
```bash
python compare_reduction_params.py
```

This will compare the reduction parameters between Python and MATLAB for Step 2 Run 2.

## Expected Output

After running, you should see:
- `upstream_matlab_log.mat` file updated
- `initReach_tracking` populated for Step 2 Run 2
- Reduction parameters (`dHmax`, `h_computed`, `redIdx`) available for comparison
