# Summary: Fixes and Root Cause Findings

## Fixes Applied

### 1. MATLAB Structure Concatenation Error ✅
**Problem**: MATLAB was throwing "Number of fields in structure arrays being concatenated do not match" errors.

**Fix**: Removed file I/O operations in `initReach_adaptive.m` that were causing structure array concatenation issues. The data is now only stored in `options.initReach_tracking` and saved to `upstreamLog`.

**Files Modified**:
- `cora_matlab/contDynamics/@linearSys/initReach_adaptive.m` (lines 88-96, 177-193)

### 2. Python Comparison Script Logic Error ✅
**Problem**: The comparison script had unreachable code due to incorrect indentation after `continue` statement.

**Fix**: Fixed the indentation so comparison logic executes correctly.

**Files Modified**:
- `compare_reduction_params.py` (lines 122-148)

## Root Cause Identified

### Time Step Divergence (Not Reduction Algorithm Bug)

**Finding**: The generator count divergence (Python: 2 vs MATLAB: 4) is **NOT** caused by the reduction algorithm, but by **different time step values** selected in the adaptive loop.

**Evidence**:
- Python Run 1: `timeStep = 0.0165` → Reduces 5→4 generators (matches MATLAB)
- Python Run 2: `timeStep = 0.0098` → Reduces 5→2 generators (differs from MATLAB)

**Impact Chain**:
```
Different timeStep 
  → Different eAt = expm(A * timeStep)
    → Different Rtrans and inputCorr
      → Different Rhom_tp generator values
        → Different reduction results (same count, different values)
```

**Key Differences Found**:
1. **Rstart centers differ**: Run 1 `[0.0267, -0.0144]` vs Run 2 `[0.0158, -0.0085]`
2. **eAt differs**: Due to different timeStep
3. **Rhom_tp generator values differ**: Max difference 0.00234 (2.6%)

**Time Step Selection**:
- Function: `_aux_optimaldeltat()` in `linReach_adaptive.py`
- Depends on: `Rstart`, `Rerror_h`, `finitehorizon`, `varphi`, `zetaP`
- Since `Rstart` differs between runs, different time steps are selected

## Next Investigation Steps

1. **Compare `_aux_optimaldeltat` inputs** between Python Run 2 and MATLAB Run 2
2. **Check if `Rstart` computation differs** between Python and MATLAB
3. **Verify `_aux_optimaldeltat` implementation** matches MATLAB exactly
4. **Compare abstraction error computation** (`Rerror_h`) between implementations

## Status

✅ **FIXED**: MATLAB structure concatenation error  
✅ **FIXED**: Python comparison script logic  
🔴 **ROOT CAUSE**: Time step divergence (not reduction algorithm bug)  
⏳ **IN PROGRESS**: Investigating why time step differs between Python Run 2 and MATLAB Run 2
