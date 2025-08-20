# Neural Network Tests

This directory contains comprehensive tests for the Neural Network module, following the structure guidelines from `readme_florian2.md`.

## Test Structure

The tests are organized according to the following pattern:
- `test_class_method` → test for `class.method`
- `test_class` → class (constructor) test
- One test file per function

## Directory Structure

```
neuralNetwork/
├── __init__.py                           # Package initialization
├── test_neuralNetwork.py                 # Constructor tests
├── test_neuralNetwork_evaluate_.py       # evaluate_ method tests
├── test_neuralNetwork_calcSensitivity.py # calcSensitivity method tests
├── test_neuralNetwork_refine.py          # refine method tests
├── test_neuralNetwork_verify.py          # verify method tests
├── test_neuralNetwork_explain.py         # explain method tests
├── test_neuralNetwork_getRefinableLayers.py # getRefinableLayers method tests
└── test_neuralNetwork_getInputNeuronOrder.py # getInputNeuronOrder method tests
```

## Test Files

### Constructor Tests (`test_neuralNetwork.py`)
- Basic constructor functionality
- Empty layers list handling
- Custom name specification
- Layer property initialization

### Method Tests
Each method has its own test file with comprehensive test coverage:

- **`test_neuralNetwork_evaluate_.py`**: Tests for the internal evaluation method
- **`test_neuralNetwork_calcSensitivity.py`**: Tests for sensitivity calculation
- **`test_neuralNetwork_refine.py`**: Tests for network refinement
- **`test_neuralNetwork_verify.py`**: Tests for formal verification
- **`test_neuralNetwork_explain.py`**: Tests for feature explanation
- **`test_neuralNetwork_getRefinableLayers.py`**: Tests for refinable layer detection
- **`test_neuralNetwork_getInputNeuronOrder.py`**: Tests for input neuron ordering

## Running Tests

### Using pytest directly
```bash
# Run all neural network tests
pytest neuralNetwork/ -v

# Run specific test file
pytest neuralNetwork/test_neuralNetwork_evaluate_.py -v

# Run specific test method
pytest neuralNetwork/test_neuralNetwork_evaluate_.py::TestNeuralNetworkEvaluate::test_evaluate_numeric_input -v
```

### Using the test runner script
```bash
# Run all tests with the provided runner
python run_neural_network_tests.py
```

### Using pytest markers
```bash
# Run only constructor tests
pytest -m constructor

# Run only evaluate method tests
pytest -m evaluate

# Run all tests except slow ones
pytest -m "not slow"
```

## Test Configuration

The `pytest.ini` file provides:
- Test discovery patterns
- Output formatting options
- Custom markers for test categorization
- Warning filters

## Dependencies

Install test dependencies with:
```bash
pip install -r requirements-test.txt
```

## Test Coverage

Each test file includes:
- **Setup methods**: Common test fixtures and mock objects
- **Basic functionality tests**: Core method behavior
- **Parameter variation tests**: Different input combinations
- **Edge case tests**: Boundary conditions and error cases
- **Integration tests**: Method interactions and dependencies

## Adding New Tests

To add tests for a new method:

1. Create a new test file following the naming convention: `test_neuralNetwork_methodName.py`
2. Follow the existing test class structure
3. Add comprehensive test coverage including edge cases
4. Update the `__init__.py` file to include the new test module
5. Add appropriate markers to `pytest.ini` if needed

## Test Data

Tests use synthetic data and mock objects to ensure:
- **Reproducibility**: Same results on different systems
- **Isolation**: Tests don't depend on external data
- **Speed**: Fast execution for development workflow
- **Coverage**: All code paths are tested

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
- Fast execution (< 30 seconds total)
- Clear error messages and stack traces
- Comprehensive coverage reporting
- Integration with pytest plugins for reporting
