"""
nnSGDOptimizer - gradient descent optimizer with optional momentum

Syntax:
    optim = nnSGDOptimizer()
    optim = nnSGDOptimizer(lr, momentum, lambda, lrDecayIter, lrDecay)

Inputs:
    lr - learning rate
    momentum - momentum
    lambda - weight decay
    lrDecayIter - iteration where learning rate is decreased
    lrDecay - learning rate decay factor

Outputs:
    optim - generated object

Reference:
    [1] https://keras.io/api/optimizers/sgd/
    [2] https://de.mathworks.com/help/deeplearning/ref/trainingoptions.html#bu80qkw-3_head

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner, Lukas Koller
Written:       01-March-2023
Last update:   27-April-2023
                25-July-2023 (LK, implemented deleteGrad)
                02-August-2023 (LK, added print function)
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, List, Optional
from .nnOptimizer import nnOptimizer


class nnSGDOptimizer(nnOptimizer):
    """
    SGD optimizer class for neural networks
    
    This class implements stochastic gradient descent with optional momentum.
    """
    
    def __init__(self, lr: float = 0.001, momentum: float = 0.9, lambda_val: float = 0, 
                 lrDecayIter: Optional[List[int]] = None, lrDecay: float = 1.0):
        """
        Constructor for nnSGDOptimizer
        
        Args:
            lr: Learning rate
            momentum: Momentum factor
            lambda_val: Weight decay factor
            lrDecayIter: Iterations where learning rate is decreased
            lrDecay: Learning rate decay factor
        """
        # parse input
        if lrDecayIter is None:
            lrDecayIter = []
        
        # validate input
        if not isinstance(momentum, (int, float)) or momentum < 0:
            raise ValueError("momentum must be a nonnegative scalar")
        
        # call super class constructor
        super().__init__(lr, lambda_val, lrDecayIter, lrDecay)
        
        self.momentum = momentum
    
    def deleteGrad(self, nn: Any, options: Dict[str, Any]) -> 'nnSGDOptimizer':
        """
        Delete gradients and reset optimizer state
        
        Args:
            nn: Neural network
            options: Options dictionary
            
        Returns:
            self: Updated optimizer
        """
        # call super class method
        super().deleteGrad(nn, options)
        
        # delete all gradients
        for i in range(len(nn)):
            layer_i = nn.layers[i]
            # Reset moment vectors
            names = layer_i.getLearnableParamNames()
            for j in range(len(names)):
                if hasattr(layer_i, 'backprop') and hasattr(layer_i.backprop, 'vel'):
                    layer_i.backprop['vel'][names[j]] = 0
        
        return self
    
    def print(self) -> str:
        """
        Print optimizer information
        
        Returns:
            s: String representation of optimizer
        """
        s = f'SGDOptimizer, Learning Rate: {self.lr:.2e}, Momentum: {self.momentum:.2e}'
        return s
    
    def updateParam(self, layer: Any, name: str, options: Optional[Dict[str, Any]] = None) -> 'nnSGDOptimizer':
        """
        Update parameter using SGD with momentum
        
        Args:
            layer: Layer to update
            name: Parameter name
            options: Update options
            
        Returns:
            self: Updated optimizer
        """
        # Read gradient
        grad = layer.backprop['grad'][name]
        
        # Read gradient velocity
        if hasattr(layer.backprop, 'vel') and name in layer.backprop['vel']:
            vel = layer.backprop['vel'][name]
        else:
            vel = 0
        
        # Update gradient velocity
        gradUpdate = self.momentum * vel - self.lr * grad
        
        # Store updated velocity
        if not hasattr(layer.backprop, 'vel'):
            layer.backprop['vel'] = {}
        layer.backprop['vel'][name] = gradUpdate
        
        # Update weight
        layer.__dict__[name] = layer.__dict__[name] + gradUpdate
        
        return self
