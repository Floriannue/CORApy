# Root Cause Analysis: Generator Count Divergence

## Summary
The divergence between Python and MATLAB starts at **Step 3's Rstart** (input to `linReach_adaptive`):
- **Python**: 2 generators
- **MATLAB**: 4 generators

This propagates through the entire computation chain, causing a 20% difference in `errorSec` and `VerrorDyn`.

## Divergence Chain

### Step 3 - Complete Chain:
1. **Rstart** (input to `linReach_adaptive`): Python 2, MATLAB 4 ❌
2. **Rdelta** (input to `initReach_adaptive`): Python 2, MATLAB 4 ❌
3. **Rhom** (before reduction): Python 15, MATLAB 21 ❌
4. **Rend.ti** (after reduction, becomes Rlinti): Python 2, MATLAB 7 ❌
5. **Rmax = Rlinti + RallError**: Python 8, MATLAB 13 ❌
6. **Rred** (after reduction): Python 2, MATLAB 4 ❌

## Key Finding

The root cause is that **Step 3's Rstart** has different generator counts. This `Rstart` comes from:
- Step 2's `Rtp` (time point reachable set)
- After reduction in `reach_adaptive` with `redFactor`

The reduction in `reach_adaptive` is:
```python
Rtp_res = Rnext['tp'].reduce('adaptive', options['redFactor'])
```

## Next Steps

1. **Compare Step 2's Rtp before reduction**: Need to track what `Rnext['tp']` is before the reduction in `reach_adaptive`
2. **Compare reduction inputs**: Verify that the input to `reduce('adaptive')` in `reach_adaptive` is the same for Python and MATLAB
3. **Fix the divergence**: Once we identify where Step 2's `Rtp` differs, fix the Python code to match MATLAB

## Files Modified for Tracking

- `cora_matlab/contDynamics/@nonlinearSys/linReach_adaptive.m` - Added `Rstart_tracking` and `Rtp_final_tracking`
- `cora_python/contDynamics/nonlinearSys/linReach_adaptive.py` - Added `Rstart_tracking` and `Rtp_final_tracking`
- `cora_matlab/contDynamics/@linearSys/initReach_adaptive.m` - Added `initReach_tracking` with `Rend_ti` and `Rend_tp`
- `cora_python/contDynamics/linearSys/initReach_adaptive.py` - Added `initReach_tracking` with `Rend_ti` and `Rend_tp`
- `cora_matlab/contDynamics/@nonlinearSys/private/priv_abstractionError_adaptive.m` - Added tracking fields to log
- `cora_python/contDynamics/nonlinearSys/private/priv_abstractionError_adaptive.py` - Added tracking fields to log

## Comparison Scripts Created

- `compare_Rstart.py` - Compares Rstart between Python and MATLAB
- `compare_initReach.py` - Compares initReach_adaptive outputs
- `compare_Rtp_final.py` - Compares final Rtp from linReach_adaptive
- `compare_reduction_inputs.py` - Compares reduction inputs and outputs
- `compare_Rlinti.py` - Compares Rlinti (linearized reachable set)
