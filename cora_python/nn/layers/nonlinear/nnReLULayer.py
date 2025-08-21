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
        if name is None:
            name = None
        # call super class constructor
        super().__init__(0, name)
    
    # evaluate ----------------------------------------------------------------
    
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
        if options['nn']['bound_approx']:
            c_ = c[j] + 0.5 * G[j, :] @ (u_ - l_)
            G_ = 0.5 * G[j, :] * np.diag(u_ - l_)
            l = c_ - np.sum(np.abs(G_))
        else:
            problem = {}
            problem['f'] = G[j, :]
            problem['Aineq'] = C
            problem['bineq'] = d
            problem['Aeq'] = []
            problem['beq'] = []
            problem['lb'] = []
            problem['ub'] = []
            # This would require CORAlinprog in MATLAB
            # For now, we'll use a simplified approach
            temp = np.min(G[j, :])
            l = c[j] + temp
        
        # compute output according to Sec. 3.2 in [1]
        if l < 0:
            # compute upper bound
            if options['nn']['bound_approx']:
                u = c_ + np.sum(np.abs(G_))
            else:
                problem = {}
                problem['f'] = -G[j, :]
                problem['Aineq'] = C
                problem['bineq'] = d
                problem['Aeq'] = []
                problem['beq'] = []
                problem['lb'] = []
                problem['ub'] = []
                # This would require CORAlinprog in MATLAB
                # For now, we'll use a simplified approach
                temp = np.max(G[j, :])
                u = c[j] - temp
            
            if u <= 0:
                # l <= u <= 0 -> linear
                c = M @ c
                G = M @ G
            else:
                # compute relaxation
                
                # constraints and offset
                C = np.vstack([
                    np.hstack([C, np.zeros((C.shape[0], 1))]),
                    np.hstack([np.zeros((1, m)), -1]),
                    np.hstack([G[j, :], -1]),
                    np.hstack([-u / (u - l) * G[j, :], 1])
                ])
                d = np.concatenate([
                    d,
                    [0],
                    [-c[j]],
                    [u / (u - l) * (c[j] - l)]
                ])
                
                # center and generators
                c = M @ c
                G = np.hstack([M @ G, self.unitvector(j, n)])
                
                # bounds
                l_ = np.concatenate([l_, [0]])
                u_ = np.concatenate([u_, [u]])
        
        return c, G, C, d, l_, u_
    
    def getMergeBuckets(self) -> int:
        """
        Get merge buckets for network reduction
        
        Returns:
            buckets: Number of merge buckets
        """
        buckets = 0
        return buckets
    
    def unitvector(self, j: int, n: int) -> np.ndarray:
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
