"""
nnOptimizer - abstract class for optimizer

Syntax:
    optim = nnOptimizer()

Inputs:
    lr - learning rate
    lambda - weight decay
    lrDecayIter - iteration where learning rate is decreased
    lrDecay - learning rate decay factor

Outputs:
    optim - generated object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner, Lukas Koller
Written:       01-March-2023
Last update:   25-July-2023 (LK, deleteGrad)
                02-August-2023 (LK, print)
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod


class nnOptimizer(ABC):
    """
    Abstract optimizer class for neural networks
    
    This class provides the interface and common functionality for all optimizers.
    Each optimizer must implement the abstract method for parameter updates.
    """
    
    def __init__(self, lr: float, lambda_val: float, lrDecayIter: List[int], lrDecay: float):
        """
        Constructor for nnOptimizer
        
        Args:
            lr: Learning rate
            lambda_val: Weight decay factor
            lrDecayIter: Iterations where learning rate is decreased
            lrDecay: Learning rate decay factor
        """
        # validate input
        if not isinstance(lr, (int, float)) or lr < 0:
            raise ValueError("lr must be a nonnegative scalar")
        if not isinstance(lambda_val, (int, float)) or lambda_val < 0:
            raise ValueError("lambda must be a nonnegative scalar")
        if not isinstance(lrDecay, (int, float)) or lrDecay < 0:
            raise ValueError("lrDecay must be a nonnegative scalar")
        
        self.lr = lr
        self.lambda_val = lambda_val
        self.lrDecayIter = lrDecayIter
        self.lrDecay = lrDecay
        
        # initialize timestep
        self.timestep = 0
    
    def step(self, nn: Any, options: Dict[str, Any], idxLayer: Optional[List[int]] = None) -> 'nnOptimizer':
        """
        Perform optimization step
        
        Args:
            nn: Neural network
            options: Options dictionary
            idxLayer: Layer indices to update (default: all layers)
            
        Returns:
            self: Updated optimizer
        """
        if idxLayer is None:
            idxLayer = list(range(len(nn)))
        
        # Increment timestep
        self.timestep = self.timestep + 1
        
        # Decrease learning rate
        if self.timestep in self.lrDecayIter:
            self.lr = self.lr * self.lrDecay
        
        # Updates all learnable parameters
        for i in idxLayer:
            layer_i = nn.layers[i]
            names = layer_i.getLearnableParamNames()
            for j in range(len(names)):
                name = names[j]
                if self.lambda_val != 0:
                    # Apply weight decay
                    layer_i.backprop['grad'][name] = \
                        layer_i.backprop['grad'][name] + self.lambda_val * layer_i.__dict__[name]
                
                self.updateParam(layer_i, name)
                
                # Clear gradient
                layer_i.backprop['grad'][name] = 0
        
        return self
    
    def deleteGrad(self, nn: Any, options: Dict[str, Any], idxLayer: Optional[List[int]] = None) -> 'nnOptimizer':
        """
        Delete gradients and reset optimizer state
        
        Args:
            nn: Neural network
            options: Options dictionary
            idxLayer: Layer indices to reset (default: all layers)
            
        Returns:
            self: Updated optimizer
        """
        if idxLayer is None:
            idxLayer = list(range(len(nn)))
        
        # reset timestep
        self.timestep = 0
        
        # delete gradients
        for i in idxLayer:
            layeri = nn.layers[i]
            # Reset backpropagation storage
            layeri.backprop['store'] = {}
            # Reset gradients
            names = layeri.getLearnableParamNames()
            for j in range(len(names)):
                layeri.backprop['grad'][names[j]] = 0
        
        return self
    
    def print(self) -> str:
        """
        Print optimizer information
        
        Returns:
            s: String representation of optimizer
        """
        s = f'Optimizer, Learning Rate: {self.lr:.2e}'
        return s
    
    @abstractmethod
    def updateParam(self, layer: Any, name: str, options: Optional[Dict[str, Any]] = None) -> 'nnOptimizer':
        """
        Abstract method to update a parameter
        
        Args:
            layer: Layer to update
            name: Parameter name
            options: Update options
            
        Returns:
            self: Updated optimizer
        """
        pass
