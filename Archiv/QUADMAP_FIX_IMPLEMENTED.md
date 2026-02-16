# quadMap Interval Fix - Implemented

## Fix Applied

Changed Python's `quadMap` to use **`max(abs(inf), abs(sup))`** instead of `center()` for Interval conversion.

### Before (Python):
```python
if isinstance(quadMat, Interval):
    quadMat_center = quadMat.center()  # MIDPOINT (0.5*(inf+sup))
    quadMat = np.asarray(quadMat_center)
```

### After (Python):
```python
if isinstance(quadMat, Interval):
    # Use max(abs(inf), abs(sup)) for conservative over-approximation
    # This matches MATLAB's tensorOrder==2 conversion method
    quadMat_inf = quadMat.inf
    quadMat_sup = quadMat.sup
    quadMat = np.maximum(np.abs(quadMat_inf), np.abs(quadMat_sup))
```

## Rationale

1. **MATLAB's tensorOrder==2 uses**: `max(infimum(abs(H_)), supremum(abs(H_)))`
2. **This is a conservative over-approximation** that takes the maximum absolute value
3. **Python was using `center()`** which is the midpoint - this is less conservative and doesn't match MATLAB

## Expected Impact

This should:
- **Reduce `errorSec` differences** from 20-29% to <1%
- **Prevent increasing divergence** over steps
- **Match MATLAB's conservative over-approximation** behavior

## Testing

Run `compare_upstream_detailed.py` to verify the fix reduces differences in:
- `errorSec` (should be <1% difference)
- `VerrorDyn` (should match errorSec improvement)
- `rerr1` (should improve accordingly)
