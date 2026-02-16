# Critical Discovery: Hessian Type Mismatch

## Key Finding

In MATLAB's `tensorOrder == 3` path:
```matlab
nlnsys = setHessian(nlnsys,'standard');  % Line 113
H = nlnsys.hessian(nlnsys.linError.p.x, nlnsys.linError.p.u);  % Line 127
errorSec = 0.5 * quadMap(Z,H);  % Line 211
```

**`setHessian('standard')` returns NUMERIC Hessian, not Interval!**

This is why `quadMap` works - it receives numeric matrices, not Interval matrices.

## Comparison

### tensorOrder == 2 (uses Interval):
```matlab
nlnsys = setHessian(nlnsys,'int');  % Interval Hessian
H = nlnsys.hessian(totalInt_x, totalInt_u);  % Returns Interval
% Then converts: H_ = max(infimum(abs(H_)), supremum(abs(H_)))
```

### tensorOrder == 3 (uses numeric):
```matlab
nlnsys = setHessian(nlnsys,'standard');  % Numeric Hessian
H = nlnsys.hessian(nlnsys.linError.p.x, nlnsys.linError.p.u);  % Returns numeric
errorSec = 0.5 * quadMap(Z,H);  % Works because H is numeric
```

## The Problem

Python might be:
1. **Returning Interval Hessian** when `setHessian('standard')` is called
2. **Not properly implementing** the 'standard' vs 'int' distinction
3. **Converting incorrectly** somewhere in the chain

## Next Steps

1. **Check Python's `setHessian('standard')`**: Does it return numeric or Interval?
2. **Check Python's `hessian()` method**: What does it return for 'standard' mode?
3. **Compare H types**: Check if Python's H is Interval when MATLAB's is numeric
4. **Fix if needed**: Ensure Python's 'standard' mode returns numeric Hessian

## Impact

If Python is using Interval Hessian when MATLAB uses numeric, this would explain:
- Why `quadMap` works in MATLAB (numeric) but Python needs conversion (Interval)
- Why the differences are 20-29% (Interval vs numeric computation)
- Why the fix didn't help (we're fixing the wrong thing!)
