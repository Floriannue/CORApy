"""
Test for nnActivationLayer abstract base class

This test verifies that the nnActivationLayer class can be imported and basic functionality works.
"""

import pytest
import numpy as np

def test_import_nnActivationLayer():
    """Test that nnActivationLayer can be imported"""
    from cora_python.nn.layers.nonlinear.nnActivationLayer import nnActivationLayer
    assert nnActivationLayer is not None

def test_nnActivationLayer_abstract():
    """Test that nnActivationLayer is abstract and cannot be instantiated directly"""
    from cora_python.nn.layers.nonlinear.nnActivationLayer import nnActivationLayer
    
    with pytest.raises(TypeError):
        # Should raise TypeError because nnActivationLayer is abstract
        nnActivationLayer()

def test_nnActivationLayer_properties():
    """Test that nnActivationLayer has the expected properties when subclassed"""
    from cora_python.nn.layers.nonlinear.nnActivationLayer import nnActivationLayer
    from cora_python.nn.layers.nnLayer import nnLayer
    
    # Create a concrete subclass for testing
    class ConcreteActivationLayer(nnActivationLayer):
        def getDf(self, i):
            if i == 0:
                return self.f
            elif i == 1:
                return lambda x: np.ones_like(x)
            else:
                return lambda x: np.zeros_like(x)
        
        def getDerBounds(self, l, u):
            # Simple implementation for testing
            return np.ones_like(l), np.ones_like(u)
        
        def evaluateNumeric(self, input_data, options):
            return input_data  # Identity function for testing
    
    layer = ConcreteActivationLayer()
    
    # Check properties
    assert layer.is_refinable == True
    assert layer.order == 1
    assert layer.do_refinement == True
    assert isinstance(layer.l, list)
    assert isinstance(layer.u, list)
    assert isinstance(layer.merged_neurons, list)
    
    # Check inheritance
    assert isinstance(layer, nnActivationLayer)
    assert isinstance(layer, nnLayer)

def test_nnActivationLayer_function_handles():
    """Test that function handles are properly initialized"""
    from cora_python.nn.layers.nonlinear.nnActivationLayer import nnActivationLayer
    
    # Create a concrete subclass for testing
    class ConcreteActivationLayer(nnActivationLayer):
        def getDf(self, i):
            if i == 0:
                return self.f
            elif i == 1:
                return lambda x: np.ones_like(x)
            else:
                return lambda x: np.zeros_like(x)
        
        def getDerBounds(self, l, u):
            # Simple implementation for testing
            return np.ones_like(l), np.ones_like(u)
        
        def evaluateNumeric(self, input_data, options):
            return input_data * 2  # Simple function for testing
    
    layer = ConcreteActivationLayer()
    
    # Test that function handles are callable
    assert callable(layer.f)
    assert callable(layer.df)
    
    # Test function evaluation
    x = np.array([[1], [2], [3]])
    result_f = layer.f(x)
    expected_f = np.array([[2], [4], [6]])  # input * 2
    assert np.allclose(result_f, expected_f)
    
    # Test derivative evaluation
    result_df = layer.df(x)
    expected_df = np.ones_like(x)
    assert np.allclose(result_df, expected_df)

def test_nnActivationLayer_getFieldStruct():
    """Test getFieldStruct method"""
    from cora_python.nn.layers.nonlinear.nnActivationLayer import nnActivationLayer
    
    # Create a concrete subclass for testing
    class ConcreteActivationLayer(nnActivationLayer):
        def __init__(self, alpha=0.5):
            super().__init__()
            self.alpha = alpha
        
        def getDf(self, i):
            return lambda x: np.ones_like(x)
        
        def getDerBounds(self, l, u):
            # Simple implementation for testing
            return np.ones_like(l), np.ones_like(u)
        
        def evaluateNumeric(self, input_data, options):
            return input_data
    
    layer = ConcreteActivationLayer(alpha=0.7)
    field_struct = layer.getFieldStruct()
    
    assert 'alpha' in field_struct
    assert field_struct['alpha'] == 0.7

def test_nnActivationLayer_getMergeBuckets():
    """Test getMergeBuckets method"""
    from cora_python.nn.layers.nonlinear.nnActivationLayer import nnActivationLayer
    
    # Create a concrete subclass for testing
    class ConcreteActivationLayer(nnActivationLayer):
        def getDf(self, i):
            return lambda x: np.ones_like(x)
        
        def getDerBounds(self, l, u):
            # Simple implementation for testing
            return np.ones_like(l), np.ones_like(u)
        
        def evaluateNumeric(self, input_data, options):
            return input_data
    
    layer = ConcreteActivationLayer()
    buckets = layer.getMergeBuckets()
    
    assert buckets == 0

def test_nnActivationLayer_abstract_methods():
    """Test that abstract methods are properly defined"""
    from cora_python.nn.layers.nonlinear.nnActivationLayer import nnActivationLayer
    
    # Create a concrete subclass that doesn't implement getDf
    class IncompleteActivationLayer(nnActivationLayer):
        def evaluateNumeric(self, input_data, options):
            return input_data
    
    with pytest.raises(TypeError):
        # Should raise TypeError because getDf is not implemented
        IncompleteActivationLayer()

def test_nnActivationLayer_getDerBounds():
    """Test getDerBounds method returns bounds within expected range"""
    from cora_python.nn.layers.nonlinear.nnActivationLayer import nnActivationLayer
    from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    from cora_python.nn.layers.nonlinear.nnTanhLayer import nnTanhLayer
    
    # Test ReLU layer
    layer = nnReLULayer()
    df_l, df_u = layer.getDerBounds(-1, 1)
    assert 0 <= df_l <= 1
    assert 0 <= df_u <= 1
    
    # Test Sigmoid layer
    layer = nnSigmoidLayer()
    df_l, df_u = layer.getDerBounds(-1, 1)
    assert 0 <= df_l <= 1
    assert 0 <= df_u <= 1
    
    # Test Tanh layer
    layer = nnTanhLayer()
    df_l, df_u = layer.getDerBounds(-1, 1)
    assert 0 <= df_l <= 1
    assert 0 <= df_u <= 1

def test_nnActivationLayer_computeApproxPoly():
    """Test computeApproxPoly method with different poly_methods and orders"""
    from cora_python.nn.layers.nonlinear.nnActivationLayer import nnActivationLayer
    from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    from cora_python.nn.layers.nonlinear.nnTanhLayer import nnTanhLayer
    
    # Test regression and ridgeregression for all activation functions
    for act, layer_class in [("relu", nnReLULayer), ("sigmoid", nnSigmoidLayer), ("tanh", nnTanhLayer)]:
        layer = layer_class()
        
        for poly_method in ["regression", "ridgeregression"]:
            for order in [1, 2, 3]:
                for n in range(1, 11):
                    # Generate random bounds
                    l = -3 + np.random.random() * 6
                    u = l + np.random.random()
                    
                    # Compute coefficients and error
                    coeffs, d = layer.computeApproxPoly(l, u, order, poly_method)
                    
                    # Evaluate polynomial
                    x = np.linspace(l, u, 100)
                    y = layer.f(x)
                    y_p = np.polyval(coeffs, x)
                    
                    # Assert containment (with tolerance for numerical errors)
                    assert np.all(y_p - d <= y + 1e-10)
                    assert np.all(y_p + d >= y - 1e-10)
    
    # Test custom poly_methods for sigmoid and tanh
    for act, layer_class in [("sigmoid", nnSigmoidLayer), ("tanh", nnTanhLayer)]:
        layer = layer_class()
        
        for poly_method in ["taylor", "throw-catch"]:
            for order in [1, 2, 3]:
                for n in range(1, 11):
                    # Generate random bounds
                    l = -3 + np.random.random() * 6
                    u = l + np.random.random()
                    
                    # Compute coefficients
                    coeffs, d = layer.computeApproxPoly(l, u, order, poly_method)
                    
                    # Evaluate polynomial
                    x = np.linspace(l, u, 100)
                    y = layer.f(x)
                    y_p = np.polyval(coeffs, x)
                    
                    # Assert containment (with tolerance for numerical errors)
                    assert np.all(y_p - d <= y + 1e-10)
                    assert np.all(y_p + d >= y - 1e-10)
    
    # Test singh method
    for act, layer_class in [("relu", nnReLULayer), ("sigmoid", nnSigmoidLayer), ("tanh", nnTanhLayer)]:
        layer = layer_class()
        poly_method = "singh"
        
        if act == "relu":
            orders = [1, 2]
        else:
            orders = [1]
        
        for order in orders:
            for n in range(1, 3):
                # Generate random bounds
                l = -3 + np.random.random() * 6
                u = l + np.random.random()
                
                # Compute coefficients
                coeffs, d = layer.computeApproxPoly(l, u, order, poly_method)
                
                # Evaluate polynomial
                x = np.linspace(l, u, 100)
                y = layer.f(x)
                y_p = np.polyval(coeffs, x)
                
                # Assert containment (with tolerance for numerical errors)
                assert np.all(y_p - d <= y + 1e-10)
                assert np.all(y_p + d >= y - 1e-10)
