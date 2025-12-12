"""
Test for neuralNetwork readONNXNetwork method

This test verifies that the readONNXNetwork method works correctly with different ONNX files.
"""

import pytest
import numpy as np
import os

def test_neuralNetwork_readONNXNetwork_basic():
    """Test readONNXNetwork with basic network"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    
    # Use available ONNX file
    modelPath = 'cora_python/examples/nn/models/ACASXU_run2a_1_2_batch_2000.onnx'
    if not os.path.exists(modelPath):
        pytest.skip("ACASXU_run2a_1_2_batch_2000.onnx file not found")
    
    # Test reading basic network
    nn = NeuralNetwork.readONNXNetwork(modelPath)
    
    # Should return a NeuralNetwork object
    assert isinstance(nn, NeuralNetwork)

def test_neuralNetwork_readONNXNetwork_verbose():
    """Test readONNXNetwork with verbose output and input/output formats"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    
    # Use available ONNX file
    modelPath = 'cora_python/examples/nn/models/ACASXU_run2a_1_2_batch_2000.onnx'
    if not os.path.exists(modelPath):
        pytest.skip("ACASXU_run2a_1_2_batch_2000.onnx file not found")
    
    # Test verbose output + input/output formats
    nn = NeuralNetwork.readONNXNetwork(modelPath, True, 'BC', 'BC')
    
    # Should return a NeuralNetwork object
    assert isinstance(nn, NeuralNetwork)

def test_neuralNetwork_readONNXNetwork_custom_layer():
    """Test readONNXNetwork with custom layer"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    
    # Use available ONNX file
    modelPath = 'cora_python/examples/nn/models/ACASXU_run2a_1_2_batch_2000.onnx'
    if not os.path.exists(modelPath):
        pytest.skip("ACASXU_run2a_1_2_batch_2000.onnx file not found")
    
    # Test reading network with custom layer
    nn = NeuralNetwork.readONNXNetwork(modelPath, False, 'BCSS')
    
    # Should return a NeuralNetwork object
    assert isinstance(nn, NeuralNetwork)

def test_neuralNetwork_readONNXNetwork_acasxu():
    """Test readONNXNetwork with ACASXU network"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    
    # Use available ONNX file
    modelPath = 'cora_python/examples/nn/models/ACASXU_run2a_1_2_batch_2000.onnx'
    if not os.path.exists(modelPath):
        pytest.skip("ACASXU_run2a_1_2_batch_2000.onnx file not found")
    
    # Test reading ACASXU network
    nn = NeuralNetwork.readONNXNetwork(modelPath, False, 'BSSC')
    
    # Should return a NeuralNetwork object
    assert isinstance(nn, NeuralNetwork)
    
    # Check that it has layers
    assert hasattr(nn, 'layers')
    assert len(nn.layers) > 0


def test_neuralNetwork_readONNXNetwork_matches_onnxruntime():
    """Ensure ONNX read produces same outputs as onnxruntime for a sample."""
    import onnxruntime as ort
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.g.macros.CORAROOT import CORAROOT
    rng = np.random.RandomState(0)

    cora_root = CORAROOT()
    modelPath = os.path.join(
        cora_root, 'cora_python', 'examples', 'nn', 'models', 'ACASXU_run2a_5_3_batch_2000.onnx'
    )
    if not os.path.exists(modelPath):
        pytest.skip(f"ACASXU model file not found: {modelPath}")

    # Load with CORA
    nn = NeuralNetwork.readONNXNetwork(modelPath, False, 'BSSC')

    # Load with onnxruntime
    sess = ort.InferenceSession(modelPath, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    # Use a deterministic sample inside the ACAS input bounds
    x = np.array([[0.64], [0.0], [0.0], [0.48], [-0.48]], dtype=np.float32)  # shape (5,1)

    # CORA evaluate (returns (5,1))
    y_cora = nn.evaluate(x)
    if isinstance(y_cora, np.ndarray):
        y_cora = y_cora.astype(np.float32)
    else:
        y_cora = np.array(y_cora, dtype=np.float32)
    y_cora = y_cora.reshape(-1)

    # ONNXRuntime evaluate (expects shape [1,1,1,5])
    x_ort = x.reshape(1, 1, 1, 5)
    y_ort = sess.run(None, {input_name: x_ort})[0].astype(np.float32).reshape(-1)

    # Compare with small tolerance
    assert np.allclose(y_cora, y_ort, rtol=1e-4, atol=1e-5), f"Mismatch: cora={y_cora}, ort={y_ort}"


@pytest.mark.parametrize("model_relpath", [
    os.path.join('cora_python', 'examples', 'nn', 'models', 'ACASXU_run2a_1_2_batch_2000.onnx'),
    os.path.join('cora_python', 'examples', 'nn', 'models', 'ACASXU_run2a_5_3_batch_2000.onnx'),
    os.path.join('cora_python', 'examples', 'nn', 'models', 'nn-nav-set.onnx'),
    os.path.join('cora_python', 'examples', 'nn', 'models', 'attitude_control_3_64_torch.onnx'),
    os.path.join('cora_python', 'examples', 'nn', 'models', 'controller_airplane.onnx'),
    os.path.join('cora_python', 'examples', 'nn', 'models', 'controller_double_pendulum_more_robust.onnx'),
    os.path.join('cora_python', 'examples', 'nn', 'models', 'controller_single_pendulum.onnx'),
    os.path.join('cora_python', 'examples', 'nn', 'models', 'controller_spacecraftDocking.onnx'),
    os.path.join('cora_python', 'examples', 'nn', 'models', 'model-cartPole.onnx'),
    os.path.join('cora_python', 'examples', 'nn', 'models', 'nn-nav-point.onnx'),
])
def test_neuralNetwork_readONNXNetwork_matches_onnxruntime_param(model_relpath):
    """Cross-check ONNX evaluation via CORA vs onnxruntime for multiple networks."""
    import onnxruntime as ort
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.g.macros.CORAROOT import CORAROOT
    rng = np.random.RandomState(1)

    cora_root = CORAROOT()
    modelPath = os.path.join(cora_root, model_relpath)
    if not os.path.exists(modelPath):
        pytest.skip(f"Model file not found: {modelPath}")

    # Load with CORA
    nn = NeuralNetwork.readONNXNetwork(modelPath)

    # Load with onnxruntime
    sess = ort.InferenceSession(modelPath, providers=['CPUExecutionProvider'])
    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    input_shape = [d if isinstance(d, int) else 1 for d in input_meta.shape]

    # Build deterministic random input matching ONNX shape
    x_ort = rng.uniform(low=-0.5, high=0.5, size=input_shape).astype(np.float32)
    # Flatten to column vector for CORA evaluate
    x_cora = x_ort.reshape(-1, 1)

    # CORA evaluate
    y_cora = nn.evaluate(x_cora)
    if isinstance(y_cora, np.ndarray):
        y_cora = y_cora.astype(np.float32)
    else:
        y_cora = np.array(y_cora, dtype=np.float32)
    y_cora = y_cora.reshape(-1)

    # ONNXRuntime evaluate
    y_ort = sess.run(None, {input_name: x_ort})[0].astype(np.float32).reshape(-1)

    # Compare with small tolerance
    assert np.allclose(y_cora, y_ort, rtol=1e-4, atol=1e-5), f"Mismatch for {model_relpath}: cora={y_cora}, ort={y_ort}"
