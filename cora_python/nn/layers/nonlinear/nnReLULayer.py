"""
nnReLULayer - class for ReLU layers

Syntax:
    obj = nnReLULayer(name)

Inputs:
    name - name of the layer, defaults to type

Outputs:
    obj - generated object

References:
    [1] Tran, H.-D., et al. "Star-Based Reachability Analysis of Deep
        Neural Networks", 2019

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork

Authors:       Tobias Ladner, Sebastian Sigl, Lukas Koller
Written:       28-March-2022
Last update:   07-July-2022 (SS, update nnLeakyReLULayer)
               04-March-2024 (LK, re-implemented evalNumeric & df_i for performance)
Last revision: 10-August-2022 (renamed)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from .nnLeakyReLULayer import nnLeakyReLULayer

class nnReLULayer(nnLeakyReLULayer):
    """
    ReLU layer class for neural networks
    
    This class implements a ReLU activation function, which is a special case of LeakyReLU
    with alpha = 0. The function computes y = max(0, x) for input x.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Constructor for nnReLULayer
        
        Args:
            name: Name of the layer, defaults to type
        """
        # call super class constructor with alpha = 0
        super().__init__(0, name)
    
    def evaluateNumeric(self, input_data: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate numeric input
        
        Args:
            input_data: Input data
            options: Evaluation options
            
        Returns:
            r: Output after ReLU activation
        """
        r = np.maximum(0, input_data)
        return r
    
    def getMergeBuckets(self) -> int:
        """
        Get merge buckets for network reduction
        
        Returns:
            buckets: Number of merge buckets
        """
        buckets = 0
        return buckets
    
    def evaluateConZonotopeNeuron(self, c: np.ndarray, G: np.ndarray, C: np.ndarray, 
                                  d: np.ndarray, l_: np.ndarray, u_: np.ndarray, 
                                  j: int, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, 
                                                                           np.ndarray, np.ndarray, 
                                                                           np.ndarray, np.ndarray]:
        """
        Evaluate constraint zonotope for a specific neuron
        
        Args:
            c: Center
            G: Generators
            C: Constraint matrix
            d: Constraint vector
            l_: Lower bounds
            u_: Upper bounds
            j: Neuron index
            options: Evaluation options
            
        Returns:
            Tuple of evaluation results
        """
        # enclose the ReLU activation function with a constrained zonotope based on
        # the results for star sets in [1]
        
        n = len(c)
        m = G.shape[1]
        M = np.eye(n)
        M[:, j] = np.zeros(n)
        
        # get lower bound
        if options.get('nn', {}).get('bound_approx', False):
            c_ = c[j] + 0.5 * G[j, :] @ (u_ - l_)
            G_ = 0.5 * G[j, :] * np.diag(u_ - l_)
            l = c_ - np.sum(np.abs(G_))
        else:
            # This would require CORAlinprog in MATLAB
            # For now, we'll use a simplified approach
            l = c[j] + np.min(G[j, :])
        
        # compute output according to Sec. 3.2 in [1]
        if l < 0:
            # compute upper bound
            if options.get('nn', {}).get('bound_approx', False):
                u = c_ + np.sum(np.abs(G_))
            else:
                # This would require CORAlinprog in MATLAB
                # For now, we'll use a simplified approach
                u = c[j] + np.max(G[j, :])
            
            if u <= 0:
                # l <= u <= 0 -> linear
                c = M @ c
                G = M @ G
            else:
                # compute relaxation
                
                # constraints and offset
                C_new = np.vstack([
                    np.hstack([C, np.zeros((C.shape[0], 1))]),
                    np.hstack([np.zeros((1, m)), -1]),
                    np.hstack([G[j, :], -1]),
                    np.hstack([-u / (u - l) * G[j, :], 1])
                ])
                
                d_new = np.concatenate([
                    d,
                    [0],
                    [-c[j]],
                    [u / (u - l) * (c[j] - l)]
                ])
                
                # center and generators
                c = M @ c
                G_new = np.hstack([M @ G, self._unitvector(j, n)])
                
                # bounds
                l_ = np.concatenate([l_, [0]])
                u_ = np.concatenate([u_, [u]])
                
                # Update variables
                C = C_new
                d = d_new
                G = G_new
        
        return c, G, C, d, l_, u_
    
    def getDf(self, i: int) -> callable:
        """
        Get derivative function
        
        Args:
            i: Derivative order
            
        Returns:
            df_i: Derivative function
        """
        if i == 0:
            return self.f
        elif i == 1:
            return lambda x: np.where(x > 0, 1, 0)
        else:
            return lambda x: np.zeros_like(x)
    
    def _unitvector(self, j: int, n: int) -> np.ndarray:
        """
        Create unit vector with 1 at position j
        
        Args:
            j: Position of 1
            n: Length of vector
            
        Returns:
            unit_vector: Unit vector
        """
        unit_vector = np.zeros(n)
        unit_vector[j] = 1
        return unit_vector.reshape(-1, 1)
