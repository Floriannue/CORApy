# Verify Test Comparison: MATLAB vs Python

## Overview
This document compares the MATLAB and Python test files for the `verify` function to ensure test coverage matches.

## Test Files

### MATLAB Test Files:
1. **`cora_matlab/unitTests/nn/neuralNetwork/testnn_neuralNetwork_verify.m`**
   - Simple test that calls example functions
   - Tests safe and unsafe verification

2. **`cora_matlab/unitTests/nn/testnn_verify.m`**
   - Comprehensive test with ACASXU models
   - Tests prop_1 and prop_2 specifications
   - Uses `aux_readNetworkAndOptions` helper

### Python Test File:
- **`cora_python/tests/nn/neuralNetwork/test_neuralNetwork_verify.py`**
   - Contains multiple test classes and functions
   - Includes both simple unit tests and comprehensive ACASXU tests

## Detailed Comparison

### 1. Simple Test (testnn_neuralNetwork_verify.m)

#### MATLAB (`testnn_neuralNetwork_verify.m`):
```matlab
function [res] = testnn_neuralNetwork_verify()
    % Use two acasxu instances from vnn-comp 2023 for testing.
    resSafe = example_neuralNetwork_verify_safe(); % Verify the specifiation.
    assert(strcmp(resSafe,'VERIFIED'));
    resUnsafe = example_neuralNetwork_verify_unsafe(); % Find a counterexample.
    assert(strcmp(resUnsafe,'COUNTEREXAMPLE'));
    res = true;
end
```

**Test Cases:**
- ✅ Calls `example_neuralNetwork_verify_safe()` - expects 'VERIFIED'
- ✅ Calls `example_neuralNetwork_verify_unsafe()` - expects 'COUNTEREXAMPLE'

#### Python (`test_neuralNetwork_verify.py`):
**Status**: ⚠️ **NOT DIRECTLY IMPLEMENTED**
- Python has `test_nn_neuralNetwork_verify_matlab_exact()` which tests with hardcoded values
- Python has `testnn_neuralNetwork_verify()` which tests ACASXU models (different from MATLAB's simple test)
- **Missing**: Direct equivalent of MATLAB's simple test that calls example functions

**Recommendation**: The Python test file has more comprehensive tests, but doesn't have the exact simple test structure. The functionality is covered by other tests.

---

### 2. Comprehensive Test (testnn_verify.m)

#### MATLAB (`testnn_verify.m`):
```matlab
function res = testnn_verify()
    modelPath = [CORAROOT '/models/Cora/nn/ACASXU_run2a_1_2_batch_2000.onnx'];
    timeout = 100;
    
    % First test case: prop_1.vnnlib
    prop1Filename = [CORAROOT '/models/Cora/nn/prop_1.vnnlib'];
    [nn,options,x,r,A,b,safeSet] = aux_readNetworkAndOptions(modelPath,prop1Filename);
    [verifRes,x_,y_] = nn.verify(x,r,A,b,safeSet,options,timeout,false);
    assert(strcmp(verifRes,'VERIFIED') & isempty(x_) & isempty(y_));
    
    % Second test case: prop_2.vnnlib
    prop2Filename = [CORAROOT '/models/Cora/nn/prop_2.vnnlib'];
    [nn,options,x,r,A,b,safeSet] = aux_readNetworkAndOptions(modelPath,prop2Filename);
    [verifRes,x_,y_] = nn.verify(x,r,A,b,safeSet,options,timeout,false);
    assert(strcmp(verifRes,'COUNTEREXAMPLE') & ~isempty(x_) & ~isempty(y_));
    assert(aux_checkCounterexample(nn,A,b,safeSet,x_,y_));
end
```

**Test Cases:**
- ✅ Test prop_1.vnnlib - expects 'VERIFIED' with empty counterexamples
- ✅ Test prop_2.vnnlib - expects 'COUNTEREXAMPLE' with non-empty counterexamples
- ✅ Validates counterexample using `aux_checkCounterexample`

#### Python (`test_neuralNetwork_verify.py` - `testnn_neuralNetwork_verify()`):
```python
def testnn_neuralNetwork_verify():
    # First test case: prop_1.vnnlib
    nn, options, x, r, A, b, safeSet = aux_readNetworkAndOptions(model1Path, prop1Filename)
    options['nn']['falsification_method'] = 'fgsm'
    options['nn']['refinement_method'] = 'naive'
    verifRes, x_, y_ = nn.verify(x, r, A, b, safeSet, options, timeout, verbose)
    assert verifRes != 'COUNTEREXAMPLE'
    
    # ... more test cases with different options ...
    
    # Second test case: prop_2.vnnlib
    nn, options, x, r, A, b, safeSet = aux_readNetworkAndOptions(model1Path, prop2Filename)
    options['nn']['falsification_method'] = 'fgsm'
    options['nn']['refinement_method'] = 'naive'
    verifRes, x_, y_ = nn.verify(x, r, A, b, safeSet, options, timeout, verbose)
    assert verifRes != 'VERIFIED'
    if verifRes == 'COUNTEREXAMPLE':
        assert aux_checkCounterexample(nn, A, b, safeSet, x_, y_)
```

**Test Cases:**
- ✅ Test prop_1.vnnlib with 'fgsm'/'naive' - expects not 'COUNTEREXAMPLE'
- ✅ Test prop_1.vnnlib with 'zonotack' - expects not 'COUNTEREXAMPLE'
- ✅ Test prop_2.vnnlib with 'fgsm'/'naive' - expects not 'VERIFIED'
- ✅ Test prop_2.vnnlib with 'zonotack' variants - expects not 'VERIFIED'
- ✅ Validates counterexample using `aux_checkCounterexample`
- ✅ **Additional**: Tests with different refinement methods and options (more comprehensive than MATLAB)

**Status**: ✅ **MORE COMPREHENSIVE** - Python test has additional test cases beyond MATLAB

---

### 3. Helper Functions Comparison

#### `aux_readNetworkAndOptions`

**MATLAB** (`testnn_verify.m` lines 56-91):
```matlab
function [nn,options,x,r,A,b,safeSet] = aux_readNetworkAndOptions(modelPath,vnnlibPath)
  options.nn = struct(...
      'use_approx_error',true,...
      'poly_method','bounds',...
      'train',struct(...
          'backprop',false,...
          'mini_batch_size',512 ...
      ) ...
  );
  options = nnHelper.validateNNoptions(options,true);
  options.nn.interval_center = false;
  nn = neuralNetwork.readONNXNetwork(modelPath,false,'BSSC');
  [X0,specs] = vnnlib2cora(vnnlibPath);
  x = 1/2*(X0{1}.sup + X0{1}.inf);
  r = 1/2*(X0{1}.sup - X0{1}.inf);
  if isa(specs.set,'halfspace')
      A = specs.set.c';
      b = -specs.set.d;
  else
      A = specs.set.A;
      b = -specs.set.b;
  end
  safeSet = strcmp(specs.type,'safeSet');
end
```

**Python** (`test_neuralNetwork_verify.py` lines 345-428):
```python
def aux_readNetworkAndOptions(modelPath: str, vnnlibPath: str):
    options = {
        'nn': {
            'use_approx_error': True,
            'poly_method': 'bounds',
            'train': {
                'backprop': False,
                'mini_batch_size': 2**8  # 256 (MATLAB: 512)
            }
        }
    }
    options = validateNNoptions(options, True)
    options['nn']['interval_center'] = False
    nn = NeuralNetwork.readONNXNetwork(modelPath, False, 'BSSC')
    X0, specs = vnnlib2cora(vnnlibPath)
    x = 0.5 * (X0[0].sup + X0[0].inf)
    r = 0.5 * (X0[0].sup - X0[0].inf)
    # ... halfspace/polytope handling ...
    safeSet = (specs.type == 'safeSet')
    return nn, options, x, r, A, b, safeSet
```

**Differences:**
- ⚠️ **mini_batch_size**: MATLAB uses `512`, Python uses `2**8 = 256` - **This is a bug! Should be 512**
- ✅ **Halfspace handling**: Python uses `representsa_` to check for halfspace (deprecated in Python)
- ⚠️ **b sign**: MATLAB uses `-specs.set.d` for halfspace (line 84), Python extracts directly - **Needs verification**

**Status**: ⚠️ **MINOR DIFFERENCES** - Batch size difference may affect test results

#### `aux_checkCounterexample`

**MATLAB** (`testnn_verify.m` lines 93-105):
```matlab
function res = aux_checkCounterexample(nn,A,b,safeSet,x_,y_)
    yi = nn.evaluate(x_);
    res = all(abs(y_ - yi) <= 1e-7,'all');
    if safeSet
        violates = any(A*yi + b >= 0,1);
    else
        violates = all(A*yi + b <= 0,1);
    end
    assert(res & violates);
end
```

**Python** (`test_neuralNetwork_verify.py` lines 297-342):
```python
def aux_checkCounterexample(nn, A, b, safeSet, x_, y_):
    yi = nn.evaluate(x_)
    res = np.all(np.abs(y_ - yi) <= 1e-7)
    if safeSet:
        violates = np.any(A @ yi >= b)
    else:
        violates = np.all(A @ yi <= b)
    assert res and violates
    return True
```

**Status**: ✅ **MATCHES** - Logic is equivalent

---

### 4. Additional Python Tests

Python has additional test cases not present in MATLAB:

#### `TestNeuralNetworkVerify` class:
- ✅ `test_verify_basic()` - Basic verification test
- ✅ `test_verify_with_timeout()` - Timeout handling
- ✅ `test_verify_with_options()` - Options handling
- ✅ `test_verify_safe_set_true()` - Safe set with True
- ✅ `test_verify_safe_set_false()` - Safe set with False
- ✅ `test_verify_none_options()` - None options handling
- ✅ `test_verify_none_timeout()` - None timeout handling
- ✅ `test_verify_verbose()` - Verbose output
- ✅ `test_verify_different_network()` - Different network instance
- ✅ `test_verify_aux_pop()` - Direct test of `_aux_pop` helper

#### `test_nn_neuralNetwork_verify_matlab_exact()`:
- ✅ Tests with hardcoded network weights matching MATLAB
- ✅ Tests both VERIFIED and COUNTEREXAMPLE cases
- ✅ Uses exact same network structure as MATLAB example

**Status**: ✅ **MORE COMPREHENSIVE** - Python has additional unit tests

---

## Summary

### Test Coverage Comparison

| Test Aspect | MATLAB | Python | Status |
|------------|--------|--------|--------|
| Simple example test | ✅ | ⚠️ (different structure) | Partial |
| ACASXU prop_1 test | ✅ | ✅ | ✅ Match |
| ACASXU prop_2 test | ✅ | ✅ | ✅ Match |
| Counterexample validation | ✅ | ✅ | ✅ Match |
| Multiple refinement methods | ❌ | ✅ | ✅ Python has more |
| Unit tests (basic functionality) | ❌ | ✅ | ✅ Python has more |
| Helper function tests | ❌ | ✅ | ✅ Python has more |

### Key Differences

1. **Test Structure:**
   - MATLAB: Two separate test files (simple and comprehensive)
   - Python: One comprehensive test file with multiple test classes

2. **Test Coverage:**
   - MATLAB: 2-3 test cases per file
   - Python: 10+ test cases with more comprehensive coverage

3. **Configuration:**
   - MATLAB `testnn_verify.m`: `timeout = 100`, `mini_batch_size = 512`
   - Python `testnn_neuralNetwork_verify()`: `timeout = 10`, `mini_batch_size = 256`

4. **Refinement Methods:**
   - MATLAB: Tests with default options only
   - Python: Tests with 'fgsm'/'naive', 'zonotack', 'zonotack-layerwise' variants

### Issues Found

1. ⚠️ **Batch size mismatch**: 
   - MATLAB: `mini_batch_size = 512`
   - Python: `mini_batch_size = 2**8 = 256`
   - **Action needed**: Update Python to use 512

2. ⚠️ **Halfspace b sign**:
   - MATLAB: `b = -specs.set.d` (line 84) - **negates d**
   - Python: `b = specs.set.b.flatten()` - **does not negate**
   - **Action needed**: Python should negate b for halfspace: `b = -specs.set.b.flatten()`

3. ⚠️ **Timeout difference**:
   - MATLAB `testnn_verify.m`: `timeout = 100`
   - Python `testnn_neuralNetwork_verify()`: `timeout = 10`
   - **Note**: This may cause different test results, but Python test is more comprehensive

### Recommendations

1. ✅ **Python tests are more comprehensive** - Good!
2. ⚠️ **Fix batch size** - Update Python to use 512 to match MATLAB
3. ⚠️ **Fix halfspace b sign** - Negate b for halfspace to match MATLAB
4. ⚠️ **Consider adding simple test** that directly calls example functions (like MATLAB's `testnn_neuralNetwork_verify.m`)
5. ✅ **Counterexample validation matches** - Good!
6. ✅ **Helper functions match** - Good!

### Conclusion

The Python test suite is **more comprehensive** than the MATLAB tests, with additional unit tests and more test cases. However, there are some configuration differences (batch size, timeout) that should be verified to ensure test results match MATLAB exactly.

