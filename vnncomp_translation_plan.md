# VNN-COMP Directory: Explanation and Translation Plan

## What is VNN-COMP?

**VNN-COMP** (Verification of Neural Networks Competition) is an annual competition for neural network verification tools. The `cora_matlab/examples/nn/vnncomp/` directory contains:
- Scripts to run CORA on VNN-COMP benchmarks
- Infrastructure for submitting CORA to the competition
- Test benchmarks for validating the verification system

## Current Directory Structure (MATLAB)

```
cora_matlab/examples/nn/vnncomp/
├── README.md                    # Documentation
├── config.yaml                  # VNN-COMP submission config
├── Dockerfile                   # Docker container for competition
├── install_tool.sh             # Installation script (bash)
├── post_install.sh             # Post-installation script (bash)
├── main_vnncomp.m              # Main entry point - runs benchmarks
├── prepare_instance.m          # Loads network & spec before verification
├── prepare_instance.sh         # Bash wrapper for prepare_instance.m
├── run_instance.m              # Runs verification for one instance
├── run_instance.sh             # Bash wrapper for run_instance.m
├── getInstanceFilename.m       # Helper to generate unique filenames
├── printErrorMessage.m         # Error reporting helper
├── scripts/
│   ├── compareResults.m        # Compare results between runs
│   ├── run_benchmarks.m        # Iterate over benchmarks
│   └── run_instances.m         # Iterate over instances in a benchmark
└── data/
    └── vnncomp2025_benchmarks/
        └── benchmarks/
            └── test/           # Test benchmark data
                ├── instances.csv
                ├── onnx/       # Neural network models
                └── vnnlib/     # Property specifications
```

## Key Files Explained

### 1. **main_vnncomp.m**
- **Purpose**: Main evaluation script for running all benchmarks
- **What it does**:
  - Sets up paths and logging
  - Defines which benchmarks to run (ACAS Xu, CIFAR100, etc.)
  - Calls `run_benchmarks()` to execute them
  - Generates results in `./results/<timestamp>/cora/`

### 2. **prepare_instance.m**
- **Purpose**: Pre-processing before the verification timer starts
- **What it does**:
  - Loads ONNX neural network using `neuralNetwork.readONNXNetwork()`
  - Loads VNNLIB specification using `vnnlib2cora()`
  - Sets benchmark-specific options (splits, constraints, GPU settings)
  - Saves everything to a .mat file for fast loading
- **Key**: Contains tuning parameters for each VNN-COMP benchmark

### 3. **run_instance.m**
- **Purpose**: Runs verification for a single (network, property) instance
- **What it does**:
  - Loads pre-processed .mat file from `prepare_instance`
  - Handles multiple input sets (batching)
  - Handles multiple specifications (union of unsafe sets)
  - Computes sensitivity-based criticality ordering
  - Calls `nn.verify()` with timeout
  - Writes result file in VNN-COMP format (sat/unsat/unknown)
  - Returns counterexample if found

### 4. **run_benchmarks.m** & **run_instances.m**
- **Purpose**: Orchestration scripts
- **What they do**:
  - Iterate over all benchmarks in `data/`
  - For each benchmark, read `instances.csv`
  - Call `prepare_instance` then `run_instance` for each
  - Collect statistics (verified/falsified/unknown)

### 5. **Bash Scripts** (.sh files)
- **Purpose**: Competition submission requirements
- **What they do**:
  - `install_tool.sh`: Installs MATLAB and CORA
  - `post_install.sh`: Restarts server for GPU drivers
  - `prepare_instance.sh`: Calls MATLAB prepare_instance.m
  - `run_instance.sh`: Calls MATLAB run_instance.m with timeout

### 6. **config.yaml**
- **Purpose**: VNN-COMP submission configuration
- **What it specifies**:
  - Tool name: CORA
  - AMI (Amazon Machine Image) for AWS
  - Script locations
  - Permissions (root/non-root)

## Translation Strategy for Python

### Approach 1: Direct Translation (Recommended for Learning/Testing)

Create equivalent Python structure:

```
cora_python/examples/nn/vnncomp/
├── README.md
├── main_vnncomp.py              # Translate main_vnncomp.m
├── prepare_instance.py          # Translate prepare_instance.m
├── run_instance.py              # Translate run_instance.m
├── get_instance_filename.py     # Translate getInstanceFilename.m
├── scripts/
│   ├── compare_results.py
│   ├── run_benchmarks.py
│   └── run_instances.py
└── data/                        # Copy from MATLAB (symlink)
    └── vnncomp2025_benchmarks/
```

**Translation Steps:**
1. Translate helper functions first:
   - `getInstanceFilename.m` → `get_instance_filename.py`
   - `printErrorMessage.m` → `print_error_message.py`

2. Translate core verification functions:
   - `prepare_instance.m` → `prepare_instance.py`
     - Use `neuralNetwork.readONNXNetwork()` (already translated)
     - Use `vnnlib2cora()` (needs translation)
     - Save to pickle instead of .mat
   
   - `run_instance.m` → `run_instance.py`
     - Use `nn.verify()` (already translated)
     - Implement criticality-based spec ordering
     - Write results in VNN-COMP format

3. Translate orchestration:
   - `run_benchmarks.m` → `run_benchmarks.py`
   - `main_vnncomp.m` → `main_vnncomp.py`


## Required Translations

### High Priority (Core Functionality):
1. ✅ **neuralNetwork.verify()** - Already translated and tested
2. ✅ **neuralNetwork.calcSensitivity()** - Already translated and fixed
3. ❌ **vnnlib2cora()** - NEEDS TRANSLATION
   - Location: Likely in `cora_matlab/nn/` or `cora_matlab/global/functions/`
   - Purpose: Parse VNN-LIB format into CORA sets
4. ❌ **Benchmark-specific options** - NEEDS TRANSLATION
   - Location: `prepare_instance.m` lines 76-361
   - Purpose: Set optimal parameters per benchmark

### Medium Priority (Competition Infrastructure):
- `run_instance.m` - Full verification pipeline
- `prepare_instance.m` - Pre-processing
- Bash scripts (.sh) - Can reuse or translate to Python

### Low Priority (Orchestration):
- `main_vnncomp.m` - Runs multiple benchmarks
- `run_benchmarks.m` - Iteration logic
- Docker/competition infrastructure

## Current Python Examples vs VNN-COMP Scripts

### What We Have:
```python
# cora_python/examples/nn/example_neuralNetwork_verify_safe.py
# - Hardcoded ACAS Xu network
# - Hardcoded property 1
# - Direct call to nn.verify()
# - Works but not flexible

# cora_python/examples/nn/example_neuralNetwork_verify_unsafe.py
# - Hardcoded ACAS Xu network
# - Hardcoded property 2
# - Direct call to nn.verify()
```

### What VNN-COMP Scripts Add:
- ✅ Load any ONNX network
- ✅ Load any VNNLIB property
- ✅ Benchmark-specific tuning
- ✅ Multiple specifications handling
- ✅ Criticality-based ordering
- ✅ Standardized output format
- ✅ Timeout handling
- ✅ Error recovery

## Recommended Translation Order

### Phase 1: Essential Dependencies (Do First)
1. Find and translate `vnnlib2cora()` function
2. Test with existing examples
3. Verify it works with prop_1.vnnlib and prop_2.vnnlib

### Phase 2: Core Verification Script
1. Create `run_instance.py` with key features:
   - Load ONNX/VNNLIB
   - Call `nn.verify()`
   - Handle multiple specs
   - Write VNN-COMP format output
2. Test with ACAS Xu benchmarks

### Phase 3: Benchmark Configuration
1. Translate benchmark options from `prepare_instance.m`
2. Create `benchmark_config.py` with options dict
3. Test with different benchmarks

### Phase 4: Full Infrastructure (Optional)
1. Translate orchestration scripts
2. Add Docker support
3. Test on full VNN-COMP suite

## Key Challenges

1. **VNNLIB Parser**: Need to find and translate `vnnlib2cora()`
2. **MAT File Format**: Replace with pickle or JSON in Python
3. **Benchmark Tuning**: 30+ benchmarks with specific options
4. **Error Handling**: MATLAB try-catch → Python try-except
5. **File Paths**: MATLAB uses `/`, Python uses `os.path` or `pathlib`
6. **Timeout**: MATLAB timeout → Python `signal.alarm()` or `threading.Timer()`

## Benefits of Translation

1. **Flexibility**: Run on any VNN-COMP benchmark
2. **Reproducibility**: Standardized verification pipeline
3. **Testing**: Validate Python implementation against competition results
4. **Competition**: Could submit Python CORA to VNN-COMP
5. **Research**: Easy experimentation with verification parameters

## Next Steps

1. **Find vnnlib2cora()**: Search MATLAB codebase
   ```bash
   grep -r "function.*vnnlib2cora" cora_matlab/
   ```

2. **Start Simple**: Create basic `verify_instance.py` script
   ```python
   # Test with existing models
   verify_instance('test', 'models/ACASXU_run2a_1_2_batch_2000.onnx', 
                   'models/prop_1.vnnlib')
   ```

3. **Incrementally Add Features**: 
   - Multi-spec support
   - Criticality ordering
   - Benchmark configs

4. **Validate Against MATLAB**: Compare results for same instances

---

**Summary**: The VNN-COMP directory is a competition submission system with benchmark orchestration, pre-processing, and standardized verification pipeline. For Python, we should:
1. First translate the core dependency (`vnnlib2cora`)
2. Create a flexible `run_instance.py` script
3. Optionally add full infrastructure later

The current Python examples already have most verification logic working - we mainly need the VNNLIB parser and better configuration management.

