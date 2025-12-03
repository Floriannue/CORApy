# Float32 Analysis and Fix Plan

## Current Float32 Usage

### 1. GPU Operations (verify.py)
**Lines 159, 174, 178, 240-242, 389, 409, 413, 424, 428, 446, 450, 643, 766, 768, 1322, 1726**

- GPU operations use `torch.float32` for performance
- This is **expected** for GPU, but CPU should use float64

### 2. Fallback to Float32 (verify_helpers.py)
**Lines 890-892, 1535-1536, 1541-1542, 1973-1974**

```python
dtype=As.dtype if As.size > 0 else np.float32  # Should be float64!
```

**Problem:** When arrays are empty, defaults to float32 instead of float64.

### 3. CastWeights Fallback (castWeights.py)
**Lines 67, 76**

```python
target_dtype = np.float32  # Should be float64!
```

**Problem:** Falls back to float32 when dtype detection fails.

### 4. ONNX Model Weights
- ONNX models typically use float32
- Need to convert to float64 after loading

## Fix Plan

### Priority 1: Fix Fallback Defaults

1. **verify_helpers.py:**
   - Change `np.float32` to `np.float64` in all fallback cases
   - Lines 890-892, 1535-1536, 1541-1542, 1973-1974

2. **castWeights.py:**
   - Change fallback from `np.float32` to `np.float64`
   - Lines 67, 76

### Priority 2: Ensure ONNX Weights are Float64

1. **readONNXNetwork.py:**
   - Convert all weights to float64 after loading
   - Or add option to cast weights to float64

2. **verify.py:**
   - Ensure inputs are float64 before verification
   - Convert ONNX weights if needed

### Priority 3: CPU vs GPU Precision

1. **verify.py:**
   - Keep GPU operations as float32 (performance)
   - Ensure CPU operations use float64
   - Convert between float32/float64 at GPU boundaries

## Implementation

### Step 1: Fix Fallback Defaults

```python
# verify_helpers.py
dtype=As.dtype if As.size > 0 else np.float64  # Changed from float32

# castWeights.py
target_dtype = np.float64  # Changed from float32
```

### Step 2: Add Float64 Conversion Helper

```python
def ensure_float64(x):
    """Ensure array is float64"""
    if x.dtype != np.float64:
        return x.astype(np.float64)
    return x
```

### Step 3: Convert ONNX Weights

```python
# In readONNXNetwork or verify
for layer in nn.layers:
    if hasattr(layer, 'W'):
        layer.W = layer.W.astype(np.float64)
    if hasattr(layer, 'b'):
        layer.b = layer.b.astype(np.float64)
```

### Step 4: Ensure Inputs are Float64

```python
# In verify function
x = x.astype(np.float64)
r = r.astype(np.float64)
A = A.astype(np.float64)
b = b.astype(np.float64)
```

## Testing

1. **Check dtype at critical points:**
   ```python
   print(f"x.dtype: {x.dtype}")
   print(f"W.dtype: {layer.W.dtype}")
   print(f"Gxi.dtype: {Gxi.dtype}")
   ```

2. **Compare with MATLAB:**
   - MATLAB uses double precision by default
   - Python should match this for CPU operations

3. **Verify no float32 in CPU path:**
   - Add assertions: `assert x.dtype == np.float64`

## Files to Modify

1. `cora_python/nn/neuralNetwork/verify_helpers.py` - Fix fallback defaults
2. `cora_python/nn/neuralNetwork/castWeights.py` - Fix fallback defaults
3. `cora_python/nn/neuralNetwork/readONNXNetwork.py` - Convert weights to float64
4. `cora_python/nn/neuralNetwork/verify.py` - Ensure inputs are float64

