# Running MATLAB Test for Comparison

## Quick Start

The MATLAB test script has been created but requires CORA's default options system. 

### Option 1: Run in MATLAB GUI

1. Open MATLAB
2. Navigate to: `D:\Bachelorarbeit\Translate_Cora`
3. Run: `test_tracking_jetEngine_matlab_complete`

### Option 2: Run from Command Line

```bash
matlab -batch "test_tracking_jetEngine_matlab_complete"
```

### Option 3: Use CORA's Example Scripts

If the test script has issues with default options, you can modify an existing CORA example:

1. Find a working CORA example that uses `reach_adaptive`
2. Add `options.traceIntermediateValues = true;` to enable tracking
3. Set `params.tFinal = 5.0;` for longer time horizon
4. Run the modified example

## Expected Output

When successful, MATLAB will create:
- `intermediate_values_step{N}_inner_loop.txt` files (one per step)
- Same format as Python trace files
- Can be compared using `compare_intermediate_values.py`

## Troubleshooting

**Error: "Unrecognized field name"**
- The test script tries to set required options, but CORA may need additional defaults
- Use CORA's `CORAoptions()` function if available
- Or modify an existing working CORA example

**No trace files created**
- Verify `options.traceIntermediateValues = true` is set
- Check that tracking code was added to MATLAB files
- Ensure write permissions in current directory

## Comparison

Once MATLAB trace files are created, compare with Python:

```bash
python compare_intermediate_values.py \
    intermediate_values_step451_inner_loop.txt \
    intermediate_values_step451_inner_loop.txt \
    1e-10
```

**Note:** Both create files with the same name. You may need to:
- Run in separate directories, OR
- Rename files before comparison
