# Float32 to Float64 Fixes Applied

## Changes Made

### 1. verify_helpers.py - Fixed Fallback Defaults

**Lines 890-892:**
- Changed `np.float32` → `np.float64` for empty array fallbacks

**Lines 1535-1536, 1541-1542:**
- Changed `np.float32` → `np.float64` for bounds fallbacks

**Lines 1973-1974:**
- Changed `np.float32` → `np.float64` for neuron bounds fallbacks

### 2. castWeights.py - Fixed Fallback Defaults

**Lines 67, 76:**
- Changed `np.float32` → `np.float64` for dtype detection fallbacks

### 3. debug_matlab_generator_collapse.m - Fixed VNNLIB Reading

**Lines 41-52:**
- Fixed to use `vnnlib2cora` instead of non-existent `neuralNetwork.readVnnlib`
- Added proper extraction of x, r, A, b, safeSet from specs

## Remaining Float32 Usage (Expected)

### GPU Operations (verify.py)
- **Lines 240-242, 389, 409, 413, 424, 428, 446, 450, 643, 766, 768:**
  - Uses `torch.float32` for GPU operations
  - **This is expected** - GPU operations use float32 for performance
  - CPU operations should use float64

### ONNX Model Weights
- ONNX models typically use float32
- Need to convert to float64 after loading (see next steps)

## Next Steps

1. **Convert ONNX weights to float64:**
   - Add conversion in `readONNXNetwork.py` or `verify.py`
   - Ensure all weights are float64 before verification

2. **Ensure inputs are float64:**
   - Add conversion at start of `verify()` function
   - Convert x, r, A, b to float64

3. **Test precision impact:**
   - Run verification with float64 everywhere
   - Compare results with MATLAB
   - Check if precision fixes generator collapse issue

## Testing

After applying fixes, verify:
```python
# Check dtypes at critical points
assert x.dtype == np.float64
assert r.dtype == np.float64
assert A.dtype == np.float64
assert b.dtype == np.float64
for layer in nn.layers:
    if hasattr(layer, 'W'):
        assert layer.W.dtype == np.float64
    if hasattr(layer, 'b'):
        assert layer.b.dtype == np.float64
```

