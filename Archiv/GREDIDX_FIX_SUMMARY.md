# gredIdx Storage Fix

## Problem Identified

In `initReach_adaptive`, the Python code was using `.append()` to store reduction indices in `gredIdx`, while MATLAB uses cell array indexing `{options.i}` to store at a specific position.

### MATLAB Code:
```matlab
[Rend.tp,~,options.gredIdx.Rhomtp{options.i}] = ...
    reduce(Rhom_tp,'adaptive',options.redFactor);
```
This stores at index `options.i` (1-based).

### Original Python Code:
```python
options['gredIdx'].setdefault('Rhomtp', []).append(idx)
```
This always appends to the end of the list.

## The Fix

Changed Python code to store at the correct index to match MATLAB:

```python
# Store at index options['i'] - 1 (0-based) to match MATLAB's {options.i} (1-based)
gredIdx = options['gredIdx'].setdefault('Rhomtp', [])
# Ensure list is long enough
while len(gredIdx) < options['i']:
    gredIdx.append(None)
gredIdx[options['i'] - 1] = idx
```

This ensures that:
- For step 1: stores at index 0 (matches MATLAB's index 1)
- For step 2: stores at index 1 (matches MATLAB's index 2)
- etc.

## Impact

This fix affects:
1. `Rhomti` reduction indices (for `Rend.ti`)
2. `Rhomtp` reduction indices (for `Rend.tp`) - **This is the source of Step 2's Rlintp divergence**
3. `Rpar` reduction indices (for `RV`)

## Status

✅ **Fixed**: Code updated to match MATLAB behavior  
⚠️ **Testing**: Test is currently failing - need to verify if this is due to the fix or a pre-existing issue

## Next Steps

1. Verify the fix doesn't break existing functionality
2. Compare Step 1's `Rend.tp` generator counts after the fix
3. Verify Step 2's `Rlintp` matches MATLAB after the fix
