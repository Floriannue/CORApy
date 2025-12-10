"""
test_nnAvgPool2DLayer - tests constructor of nnAvgPool2DLayer

Syntax:
    pytest test_nnAvgPool2DLayer.py

Inputs:
    -

Outputs:
    Test results

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner
Written:       02-October-2023
Last update:   ---
Last revision: ---

Translated to Python by: Florian Nüssel
Translation date: 2025-11-25
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from cora_python.nn.layers.linear.nnAvgPool2DLayer import nnAvgPool2DLayer
from cora_python.nn.neuralNetwork import NeuralNetwork


def test_nnAvgPool2DLayer():
    """
    Test constructor and evaluation of nnAvgPool2DLayer
    """
    # Simple example - test constructor
    layer = nnAvgPool2DLayer([2, 2])
    # Convert torch tensors to numpy for comparison
    W_np = layer.W.cpu().numpy() if hasattr(layer.W, 'cpu') else layer.W
    stride_np = layer.stride.cpu().numpy() if hasattr(layer.stride, 'cpu') else layer.stride
    assert np.allclose(W_np, 0.25 * np.ones((2, 2))), "Weight matrix should be 0.25 * ones(2,2)"
    assert np.array_equal(stride_np, [2, 2]), "Stride should be [2, 2]"
    
    layer = nnAvgPool2DLayer([1, 1])
    # Convert torch tensors to numpy for comparison
    W_np = layer.W.cpu().numpy() if hasattr(layer.W, 'cpu') else layer.W
    stride_np = layer.stride.cpu().numpy() if hasattr(layer.stride, 'cpu') else layer.stride
    assert np.allclose(W_np, 1), "Weight matrix should be 1"
    assert np.array_equal(stride_np, [1, 1]), "Stride should be [1, 1]"
    
    # Check evaluate
    layer = nnAvgPool2DLayer([2, 2])
    nn = NeuralNetwork([layer])
    n = 4
    nn.setInputSize([n, n, 1])
    
    # Check point evaluation
    x = np.reshape(np.eye(n), (-1, 1))
    y = nn.evaluate(x)
    y_true = np.array([0.5, 0, 0, 0.5]).reshape(-1, 1)
    
    assert np.allclose(y, y_true), f"Output mismatch: expected {y_true.flatten()}, got {y.flatten()}"
    
    # Check zonotope evaluation (if zonotope class is available)
    try:
        from cora_python.contSet.zonotope.zonotope import zonotope
        X = zonotope(x, 0.01 * np.eye(n * n))
        Y = nn.evaluate(X)
        
        # Check if Y contains y
        assert Y.contains(y), "Zonotope output should contain point output"
    except ImportError:
        # Zonotope class not available, skip this test
        pass
    
    print("test_nnAvgPool2DLayer passed!")


def test_nnAvgPool2DLayer_matlab_validation():
    """
    Test AvgPool2D evaluation against MATLAB output from vnn_verivital_avgpool.onnx
    Uses exact MATLAB values for validation
    """
    import os
    from cora_python.g.macros.CORAROOT import CORAROOT
    
    # Load the actual network from ONNX file
    model_path = os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'unitTests', 'vnn_verivital_avgpool.onnx')
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found: {model_path}")
    
    nn = NeuralNetwork.readONNXNetwork(model_path, False, 'BCSS')
    
    # Test with all-ones input (matches MATLAB test)
    x = np.ones((784, 1))
    
    # Evaluate through network (first 3 layers: Conv2D + ReLU + AvgPool2D)
    options = {'nn': {}}
    y_avgpool = nn.evaluate_(x, options, [0, 1, 2])  # 0-based indexing
    
    # MATLAB expected output (from matlab_conv_avgpool_output.txt line 37)
    # First 36 values should be 0.500804, then zeros
    expected_first = 0.500804
    expected_zeros = 0.0
    tol = 1e-5
    
    assert y_avgpool.shape == (1152, 1), f"Expected shape (1152, 1), got {y_avgpool.shape}"
    
    # Check first 36 values (should be ~0.500804)
    assert np.allclose(y_avgpool[:36], expected_first, atol=tol), \
        f"First 36 values should be ~{expected_first}, got {y_avgpool[:36].flatten()[:10]}"
    
    # Check values after index 36 (should be zeros after ReLU)
    assert np.allclose(y_avgpool[36:50], expected_zeros, atol=tol), \
        f"Values from index 36 should be ~0, got {y_avgpool[36:50].flatten()[:10]}"
    
    # Check last 10 values (should be zeros)
    assert np.allclose(y_avgpool[-10:], expected_zeros, atol=tol), \
        f"Last 10 values should be ~0, got {y_avgpool[-10:].flatten()}"


def test_nnAvgPool2DLayer_random_input_matlab_validation():
    """
    Test AvgPool2D evaluation with random input against MATLAB output
    Uses exact MATLAB random input values (from matlab_conv_avgpool_output.txt line 43)
    """
    import os
    from cora_python.g.macros.CORAROOT import CORAROOT
    
    # Load the actual network from ONNX file
    model_path = os.path.join(CORAROOT(), 'cora_matlab', 'models', 'Cora', 'nn', 'unitTests', 'vnn_verivital_avgpool.onnx')
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found: {model_path}")
    
    nn = NeuralNetwork.readONNXNetwork(model_path, False, 'BCSS')
    
    # Exact MATLAB random input values (784 values from matlab_conv_avgpool_output.txt)
    # These are the exact values MATLAB used with rng('default')
    x_rand_values = np.array([
        0.814724, 0.905792, 0.126987, 0.913376, 0.632359, 0.097540, 0.278498, 0.546882, 0.957507, 0.964889,
        0.157613, 0.970593, 0.957167, 0.485376, 0.800280, 0.141886, 0.421761, 0.915736, 0.792207, 0.959492,
        0.655741, 0.035712, 0.849129, 0.933993, 0.678735, 0.757740, 0.743132, 0.392227, 0.655478, 0.171187,
        0.706046, 0.031833, 0.276923, 0.046171, 0.097132, 0.823458, 0.694829, 0.317099, 0.950222, 0.034446,
        0.438744, 0.381558, 0.765517, 0.795200, 0.186873, 0.489764, 0.445586, 0.646313, 0.709365, 0.754687,
        0.276025, 0.679703, 0.655098, 0.162612, 0.118998, 0.498364, 0.959744, 0.340386, 0.585268, 0.223812,
        0.751267, 0.255095, 0.505957, 0.699077, 0.890903, 0.959291, 0.547216, 0.138624, 0.149294, 0.257508,
        0.840717, 0.254282, 0.814285, 0.243525, 0.929264, 0.349984, 0.196595, 0.251084, 0.616045, 0.473289,
        0.351660, 0.830829, 0.585264, 0.549724, 0.917194, 0.285839, 0.757200, 0.753729, 0.380446, 0.567822,
        0.075854, 0.053950, 0.530798, 0.779167, 0.934011, 0.129906, 0.568824, 0.469391, 0.011902, 0.337123,
        0.162182, 0.794285, 0.311215, 0.528533, 0.165649, 0.601982, 0.262971, 0.654079, 0.689215, 0.748152,
        0.450542, 0.083821, 0.228977, 0.913337, 0.152378, 0.825817, 0.538342, 0.996135, 0.078176, 0.442678,
        0.106653, 0.961898, 0.004634, 0.774910, 0.817303, 0.868695, 0.084436, 0.399783, 0.259870, 0.800068,
        0.431414, 0.910648, 0.181847, 0.263803, 0.145539, 0.136069, 0.869292, 0.579705, 0.549860, 0.144955,
        0.853031, 0.622055, 0.350952, 0.513250, 0.401808, 0.075967, 0.239916, 0.123319, 0.183908, 0.239953,
        0.417267, 0.049654, 0.902716, 0.944787, 0.490864, 0.489253, 0.337719, 0.900054, 0.369247, 0.111203,
        0.780252, 0.389739, 0.241691, 0.403912, 0.096455, 0.131973, 0.942051, 0.956135, 0.575209, 0.059780,
        0.234780, 0.353159, 0.821194, 0.015403, 0.043024, 0.168990, 0.649115, 0.731722, 0.647746, 0.450924,
        0.547009, 0.296321, 0.744693, 0.188955, 0.686775, 0.183511, 0.368485, 0.625619, 0.780227, 0.081126,
        0.929386, 0.775713, 0.486792, 0.435859, 0.446784, 0.306349, 0.508509, 0.510772, 0.817628, 0.794831,
        0.644318, 0.378609, 0.811580, 0.532826, 0.350727, 0.939002, 0.875943, 0.550156, 0.622475, 0.587045,
        0.207742, 0.301246, 0.470923, 0.230488, 0.844309, 0.194764, 0.225922, 0.170708, 0.227664, 0.435699,
        0.311102, 0.923380, 0.430207, 0.184816, 0.904881, 0.979748, 0.438870, 0.111119, 0.258065, 0.408720,
        0.594896, 0.262212, 0.602843, 0.711216, 0.221747, 0.117418, 0.296676, 0.318778, 0.424167, 0.507858,
        0.085516, 0.262482, 0.801015, 0.029220, 0.928854, 0.730331, 0.488609, 0.578525, 0.237284, 0.458849,
        0.963089, 0.546806, 0.521136, 0.231594, 0.488898, 0.624060, 0.679136, 0.395515, 0.367437, 0.987982,
        0.037739, 0.885168, 0.913287, 0.796184, 0.098712, 0.261871, 0.335357, 0.679728, 0.136553, 0.721227,
        0.106762, 0.653757, 0.494174, 0.779052, 0.715037, 0.903721, 0.890923, 0.334163, 0.698746, 0.197810,
        0.030541, 0.744074, 0.500022, 0.479922, 0.904722, 0.609867, 0.617666, 0.859442, 0.805489, 0.576722,
        0.182922, 0.239932, 0.886512, 0.028674, 0.489901, 0.167927, 0.978681, 0.712694, 0.500472, 0.471088,
        0.059619, 0.681972, 0.042431, 0.071445, 0.521650, 0.096730, 0.818149, 0.817547, 0.722440, 0.149865,
        0.659605, 0.518595, 0.972975, 0.648991, 0.800331, 0.453798, 0.432392, 0.825314, 0.083470, 0.133171,
        0.173389, 0.390938, 0.831380, 0.803364, 0.060471, 0.399258, 0.526876, 0.416799, 0.656860, 0.627973,
        0.291984, 0.431651, 0.015487, 0.984064, 0.167168, 0.106216, 0.372410, 0.198118, 0.489688, 0.339493,
        0.951630, 0.920332, 0.052677, 0.737858, 0.269119, 0.422836, 0.547871, 0.942737, 0.417744, 0.983052,
        0.301455, 0.701099, 0.666339, 0.539126, 0.698106, 0.666528, 0.178132, 0.128014, 0.999080, 0.171121,
        0.032601, 0.561200, 0.881867, 0.669175, 0.190433, 0.368917, 0.460726, 0.981638, 0.156405, 0.855523,
        0.644765, 0.376272, 0.190924, 0.428253, 0.482022, 0.120612, 0.589507, 0.226188, 0.384619, 0.582986,
        0.251806, 0.290441, 0.617091, 0.265281, 0.824376, 0.982663, 0.730249, 0.343877, 0.584069, 0.107769,
        0.906308, 0.879654, 0.817761, 0.260728, 0.594356, 0.022513, 0.425259, 0.312719, 0.161485, 0.178766,
        0.422886, 0.094229, 0.598524, 0.470924, 0.695949, 0.699888, 0.638531, 0.033604, 0.068806, 0.319600,
        0.530864, 0.654446, 0.407619, 0.819981, 0.718359, 0.968649, 0.531334, 0.325146, 0.105629, 0.610959,
        0.778802, 0.423453, 0.090823, 0.266471, 0.153657, 0.281005, 0.440085, 0.527143, 0.457424, 0.875372,
        0.518052, 0.943623, 0.637709, 0.957694, 0.240707, 0.676122, 0.289065, 0.671808, 0.695140, 0.067993,
        0.254790, 0.224040, 0.667833, 0.844392, 0.344462, 0.780520, 0.675332, 0.006715, 0.602170, 0.386771,
        0.915991, 0.001151, 0.462449, 0.424349, 0.460916, 0.770160, 0.322472, 0.784739, 0.471357, 0.035763,
        0.175874, 0.721758, 0.473486, 0.152721, 0.341125, 0.607389, 0.191745, 0.738427, 0.242850, 0.917424,
        0.269062, 0.765500, 0.188662, 0.287498, 0.091113, 0.576209, 0.683363, 0.546593, 0.425729, 0.644443,
        0.647618, 0.679017, 0.635787, 0.945174, 0.208935, 0.709282, 0.236231, 0.119396, 0.607304, 0.450138,
        0.458725, 0.661945, 0.770286, 0.350218, 0.662010, 0.416159, 0.841929, 0.832917, 0.256441, 0.613461,
        0.582249, 0.540739, 0.869941, 0.264779, 0.318074, 0.119215, 0.939829, 0.645552, 0.479463, 0.639317,
        0.544716, 0.647311, 0.543886, 0.721047, 0.522495, 0.993705, 0.218677, 0.105798, 0.109697, 0.063591,
        0.404580, 0.448373, 0.365816, 0.763505, 0.627896, 0.771980, 0.932854, 0.972741, 0.192028, 0.138874,
        0.696266, 0.093820, 0.525404, 0.530344, 0.861140, 0.484853, 0.393456, 0.671431, 0.741258, 0.520052,
        0.347713, 0.149997, 0.586092, 0.262145, 0.044454, 0.754933, 0.242785, 0.442402, 0.687796, 0.359228,
        0.736340, 0.394707, 0.683416, 0.704047, 0.442305, 0.019578, 0.330858, 0.424309, 0.270270, 0.197054,
        0.821721, 0.429921, 0.887771, 0.391183, 0.769114, 0.396792, 0.808514, 0.755077, 0.377396, 0.216019,
        0.790407, 0.949304, 0.327565, 0.671264, 0.438645, 0.833501, 0.768854, 0.167254, 0.861980, 0.989872,
        0.514423, 0.884281, 0.588026, 0.154752, 0.199863, 0.406955, 0.748706, 0.825584, 0.789963, 0.318524,
        0.534064, 0.089951, 0.111706, 0.136293, 0.678652, 0.495177, 0.189710, 0.495006, 0.147608, 0.054974,
        0.850713, 0.560560, 0.929609, 0.696667, 0.582791, 0.815397, 0.879014, 0.988912, 0.000522, 0.865439,
        0.612566, 0.989950, 0.527680, 0.479523, 0.801348, 0.227843, 0.498094, 0.900852, 0.574661, 0.845178,
        0.738640, 0.585987, 0.246735, 0.666416, 0.083483, 0.625960, 0.660945, 0.729752, 0.890752, 0.982303,
        0.769029, 0.581446, 0.928313, 0.580090, 0.016983, 0.120860, 0.862711, 0.484297, 0.844856, 0.209405,
        0.552291, 0.629883, 0.031991, 0.614713, 0.362411, 0.049533, 0.489570, 0.192510, 0.123084, 0.205494,
        0.146515, 0.189072, 0.042652, 0.635198, 0.281867, 0.538597, 0.695163, 0.499116, 0.535801, 0.445183,
        0.123932, 0.490357, 0.852998, 0.873927, 0.270294, 0.208461, 0.564980, 0.640312, 0.417029, 0.205976,
        0.947933, 0.082071, 0.105709, 0.142041, 0.166460, 0.620959, 0.573710, 0.052078, 0.931201, 0.728662,
        0.737842, 0.063405, 0.860441, 0.934405, 0.984398, 0.858939, 0.785559, 0.513377, 0.177602, 0.398589,
        0.133931, 0.030890, 0.939142, 0.301306, 0.295534, 0.332936, 0.467068, 0.648198, 0.025228, 0.842207,
        0.559033, 0.854100, 0.347879, 0.446027, 0.054239, 0.177108, 0.662808, 0.330829, 0.898486, 0.118155,
        0.988418, 0.539982, 0.706917, 0.999492, 0.287849, 0.414523, 0.464840, 0.763957, 0.818204, 0.100222,
        0.178117, 0.359635, 0.056705, 0.521886, 0.335849, 0.175669, 0.208947, 0.905154, 0.675391, 0.468468,
        0.912132, 0.104012, 0.745546, 0.736267, 0.561861, 0.184194, 0.597211, 0.299937, 0.134123, 0.212602,
        0.894942, 0.071453, 0.242487, 0.053754, 0.441722, 0.013283, 0.897191, 0.196658, 0.093371, 0.307367,
        0.456058, 0.101669, 0.995390, 0.332093, 0.297347, 0.062045, 0.298244, 0.046351, 0.505428, 0.761426,
        0.631070, 0.089892, 0.080862, 0.777241, 0.905135, 0.533772, 0.109154, 0.825809, 0.338098, 0.293973,
        0.746313, 0.010337, 0.048447, 0.667916, 0.603468, 0.526102, 0.729709, 0.707253, 0.781377, 0.287977,
        0.692532, 0.556670, 0.396521, 0.061591
    ])
    
    x_rand = x_rand_values.reshape(784, 1)
    
    # Evaluate through network (first 3 layers: Conv2D + ReLU + AvgPool2D)
    options = {'nn': {}}
    y_avgpool_rand = nn.evaluate_(x_rand, options, [0, 1, 2])
    
    # MATLAB expected output (first 50 values from line 46)
    matlab_expected = np.array([
        0.053689, 0.031656, 0.092406, 0.040831, 0.044133, 0.035847, 0.022120, 0.002673, 
        0.042199, 0.046572, 0.013991, 0.002218, 0.024745, 0.103493, 0.050126, 0.036364, 
        0.009492, 0.058605, 0.017426, 0.029114, 0.029997, 0.032772, 0.011409, 0.089245, 
        0.041041, 0.053843, 0.022806, 0.007909, 0.042878, 0.083852, 0.025088, 0.004915, 
        0.044409, 0.132317, 0.077495, 0.058315, 0.000000, 0.000000, 0.000000, 0.000000, 
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 
        0.000000, 0.000000
    ])
    
    tol = 1e-4  # Allow for numerical differences
    assert y_avgpool_rand.shape == (1152, 1), f"Expected shape (1152, 1), got {y_avgpool_rand.shape}"
    assert np.allclose(y_avgpool_rand[:50].flatten(), matlab_expected, atol=tol), \
        f"First 50 values don't match MATLAB. Got {y_avgpool_rand[:50].flatten()[:10]}, expected {matlab_expected[:10]}"


if __name__ == '__main__':
    test_nnAvgPool2DLayer()
    test_nnAvgPool2DLayer_matlab_validation()
    test_nnAvgPool2DLayer_random_input_matlab_validation()
    print("All tests passed!")


# ------------------------------ END OF CODE ------------------------------

