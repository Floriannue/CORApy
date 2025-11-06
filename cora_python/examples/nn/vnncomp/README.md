# CORA Python VNN-COMP Scripts

This folder contains Python scripts for running CORA on VNN-COMP benchmarks.

## Files

- `get_instance_filename.py`: Helper to generate unique filenames for instances
- `run_instance.py`: Main verification script for a single instance
- `scripts/run_benchmarks.py`: Orchestration for running multiple benchmarks

## Usage

### Run a single instance

```python
python run_instance.py <benchmark_name> <onnx_path> <vnnlib_path> <results_path> <timeout> [--verbose]
```

Example:
```python
python run_instance.py test models/ACASXU_run2a_1_2_batch_2000.onnx models/prop_1.vnnlib results.txt 120 --verbose
```

### Usage Example
```bash
cd cora_python/examples/nn/vnncomp
export PYTHONPATH="/path/to/Translate_Cora"
python run_instance.py test \
    ../models/ACASXU_run2a_1_2_batch_2000.onnx \
    ../models/prop_2.vnnlib \
    results.txt \
    120 \
    --verbose
```

### Output format

Results are written in VNN-COMP format:
- `unsat` - Property verified (safe)
- `sat` - Counterexample found (unsafe) with witness values
- `unknown` - Could not determine within timeout

## Differences from MATLAB

1. Uses pickle instead of .mat files for caching
2. Python-native argument parsing
3. No separate `prepare_instance` step (integrated into `run_instance`)

## Requirements

- Python 3.11+
- CORA Python (cora_python package)
- (Optional) GPU support for performance

---

**Translation**: Florian Nüssel, BA 2025
Based on MATLAB VNN-COMP submission by Lukas Koller, 2025

