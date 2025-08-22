"""
lookupDf - look-up table for i-th derivative of the given layer
   global look-up table to reuse the computation of the derivative

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Tobias Ladner
         Python: AI Assistant
"""

from typing import Any, Dict, Callable
import numpy as np
import sympy as sp
from sympy import symbols, diff


# Global lookup dictionary (equivalent to MATLAB's persistent variable)
_lookupDf: Dict[str, Dict[int, Any]] = {}


def lookupDf(layer: Any, i: int) -> Callable:
    """
    Look-up table for i-th derivative of the given layer.
    Global look-up table to reuse the computation of the derivative.
    
    Args:
        layer: nnActivationLayer
        i: i-th derivative
        
    Returns:
        df_i: function handle of the i-th derivative
        
    See also: -
    """
    global _lookupDf
    
    # find layer name
    layerName = layer.__class__.__name__
    
    # check if present
    if layerName not in _lookupDf:
        # init layer struct
        layerStruct = {}
        
        # init first derivative
        # Use sympy for symbolic differentiation
        if hasattr(layer, 'f'):
            # Create symbolic variable
            x = symbols('x')
            
            # Get the symbolic function from the layer
            if hasattr(layer, 'getSymbolicFunction'):
                f_sym = layer.getSymbolicFunction(x)
            else:
                # Fallback: try to create symbolic function from the layer's f method
                # This is a simplified approach - in practice, layers should implement getSymbolicFunction
                f_sym = _create_symbolic_function(layer, x)
            
            # Compute first derivative symbolically
            df1_sym = diff(f_sym, x)
            
            # Convert to lambda function for evaluation
            df1_lambda = sp.lambdify(x, df1_sym, modules=['numpy'])
            
            layerStruct[1] = {'df': df1_lambda}
        else:
            # Fallback: create a simple numerical derivative
            def numerical_df(x):
                h = 1e-8
                return (layer.f(x + h) - layer.f(x - h)) / (2 * h)
            layerStruct[1] = {'df': numerical_df}
        
        # store
        _lookupDf[layerName] = layerStruct
    
    # read layer struct
    layerStruct = _lookupDf[layerName]
    
    # returns a function handle of the i-th derivative of this layer
    if i == 0:
        # function
        df_i = layer.f
    
    elif i <= len(layerStruct):
        # derivative is known
        df_i = layerStruct[i]['df']
    
    else:
        # find derivative
        maxKnownDer = len(layerStruct)
        
        # find last known derivative
        df_j1 = layerStruct[maxKnownDer]['df']
        
        # iterate up to desired derivative
        for j in range(maxKnownDer + 1, i + 1):
            # compute symbolic derivative
            if hasattr(layer, 'getSymbolicFunction'):
                x = symbols('x')
                f_sym = layer.getSymbolicFunction(x)
                
                # Compute j-th derivative symbolically
                df_j_sym = f_sym
                for _ in range(j):
                    df_j_sym = diff(df_j_sym, x)
                
                # Convert to lambda function
                df_j_lambda = sp.lambdify(x, df_j_sym, modules=['numpy'])
                layerStruct[j] = {'df': df_j_lambda}
            else:
                # Fallback: numerical differentiation
                def numerical_df_j(x):
                    h = 1e-8
                    return (df_j1(x + h) - df_j1(x - h)) / (2 * h)
                
                layerStruct[j] = {'df': numerical_df_j}
            
            # prepare for next iteration
            df_j1 = layerStruct[j]['df']
        
        # retrieve result
        df_i = df_j1
        
        # store result
        _lookupDf[layerName] = layerStruct
    
    return df_i


def _create_symbolic_function(layer: Any, x: sp.Symbol) -> sp.Expr:
    """
    Create a symbolic function from the layer's f method.
    This is a fallback method when the layer doesn't provide getSymbolicFunction.
    
    Args:
        layer: neural network layer
        x: symbolic variable
        
    Returns:
        f_sym: symbolic function expression
    """
    # This is a simplified approach - in practice, layers should implement
    # getSymbolicFunction to provide their exact symbolic representation
    
    # Try to identify common activation functions
    if hasattr(layer, 'activation') and layer.activation is not None:
        activation_name = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
        
        if 'relu' in activation_name.lower():
            return sp.Max(0, x)
        elif 'sigmoid' in activation_name.lower() or 'logistic' in activation_name.lower():
            return 1 / (1 + sp.exp(-x))
        elif 'tanh' in activation_name.lower():
            return sp.tanh(x)
        elif 'exp' in activation_name.lower():
            return sp.exp(x)
        elif 'log' in activation_name.lower():
            return sp.log(x)
        elif 'sin' in activation_name.lower():
            return sp.sin(x)
        elif 'cos' in activation_name.lower():
            return sp.cos(x)
        else:
            # Unknown activation function, return identity
            return x
    
    # Default: return identity function
    return x


def clear_lookup_table():
    """
    Clear the global lookup table.
    Useful for testing or when memory usage becomes an issue.
    """
    global _lookupDf
    _lookupDf.clear()


def get_lookup_table_info() -> Dict[str, Any]:
    """
    Get information about the current lookup table state.
    
    Returns:
        info: dictionary with lookup table information
    """
    global _lookupDf
    
    info = {}
    for layer_name, layer_struct in _lookupDf.items():
        info[layer_name] = {
            'known_derivatives': list(layer_struct.keys()),
            'max_derivative': max(layer_struct.keys()) if layer_struct else 0
        }
    
    return info
