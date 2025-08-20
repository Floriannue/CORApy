"""
nnLayer - abstract class for nn layers

Syntax:
    nnLayer

Inputs:
    name - name of the layer, defaults to type

Outputs:
    obj - generated object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork

Authors:       Tobias Ladner
Written:       28-March-2022
Last update:   15-February-2023 (re-organized, simplified)
               24-February-2023 (added options to all evaluate func)
               31-July-2023 (LK, modified 'backprop' signature)
               02-August-2023 (LK, zonotope batch-eval & -backprop)
               22-January-2022 (LK, functions for IBP-based training)
               24-February-2023 (added options to all evaluate func)
               18-August-2024 (MW, updateGradient -> policy grad)
Last revision: 10-August-2022 (renamed)
               Automatic python translation: Florian Nüssel BA 2025
"""

from abc import ABC, abstractmethod
import copy
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from cora_python.contSet.interval import Interval

class nnLayer(ABC):
    """
    Abstract base class for neural network layers
    
    This class provides the interface and common functionality for all neural network layers.
    Each layer must implement the abstract methods for evaluation and neuron counting.
    """
    
    # Class variable for unique naming
    _count = 0
    
    def __init__(self, name: Optional[str] = None):
        """
        Constructor for nnLayer
        
        Args:
            name: Name of the layer, defaults to type
        """
        if name is None or name == "":
            name = self.getDefaultName()
        
        self.name = name
        self.inputSize = []
        
        # sensitivity of input
        self.sensitivity = []
        
        # for backpropagation
        self.backprop = {'store': {}}
    
    def computeSizes(self, inputSize: List[int]) -> List[int]:
        """
        Compute and set the input size, return output size
        
        Args:
            inputSize: Input dimensions
            
        Returns:
            outputSize: Output dimensions
        """
        self.inputSize = inputSize
        outputSize = self.getOutputSize(inputSize)
        return outputSize
    
    def getDefaultName(self) -> str:
        """
        Get default name from class name
        
        Returns:
            name: Default name for the layer
        """
        # get name from class name 
        name = self.__class__.__name__
        if name.startswith('nn'):
            name = name[3:]
        name = name.replace('Layer', '')
        
        # add unique number to name
        name = f"{name}_{nnLayer._getCount()}"
        return name
    
    def getLayerInfo(self) -> str:
        """
        Get string representation of layer information
        
        Returns:
            str: Formatted layer information
        """
        str_info = f"{self.__class__.__name__:<30} {self.name:<30}"
        
        if self.inputSize:
            input_str = 'x'.join(map(str, self.inputSize))
            output = self.getOutputSize(self.inputSize)
            output_str = 'x'.join(map(str, output))
            
            str_info += f" {input_str:>10} -> {output_str:<10}"
        else:
            str_info += "\t(Input size not set)"
        
        # additional information
        if hasattr(self, 'type') and hasattr(self, '__class__'):
            if 'GNNGlobalPooling' in self.__class__.__name__:
                str_info += f'(pooling across nodes ({self.type}))'
            elif 'GNN' in self.__class__.__name__:
                str_info += "(per node)"
        
        return str_info
    
    def evaluate(self, *args, **kwargs):
        """
        Wrapper to propagate a single layer
        
        Args:
            *args: Variable arguments for evaluation
            **kwargs: Keyword arguments for evaluation
            
        Returns:
            r: Evaluation result
        """
        # Create a minimal neural network with just this layer
        from cora_python.nn.neuralNetwork import NeuralNetwork
        nn = NeuralNetwork([self])
        r = nn.evaluate(*args, **kwargs)
        return r
    
    def castWeights(self, x):
        """
        Callback when data type of learnable parameters was changed
        
        Args:
            x: New data type
        """
        # Default implementation does nothing
        pass
    
    @abstractmethod
    def evaluateNumeric(self, input_data: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Abstract method for numeric evaluation
        
        Args:
            input_data: Input data
            options: Evaluation options
            
        Returns:
            r: Numeric evaluation result
        """
        pass
    
    def evaluateSensitivity(self, S: np.ndarray, x: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate sensitivity (default implementation throws error)
        
        Args:
            S: Sensitivity matrix
            x: Input point
            options: Evaluation options
            
        Returns:
            S: Sensitivity result
            
        Raises:
            NotImplementedError: If sensitivity is not supported
        """
        raise NotImplementedError(f"Sensitivity evaluation not supported for {self.__class__.__name__}")
    
    def evaluateInterval(self, bounds: 'Interval', options: Dict[str, Any]) -> 'Interval':
        """
        Evaluate interval (default implementation uses numeric evaluation)
        
        Args:
            bounds: Input interval bounds
            options: Evaluation options
            
        Returns:
            bounds: Output interval bounds
        """

        
        # Convert interval to numeric bounds and evaluate
        bounds_numeric = np.stack([bounds.inf, bounds.sup], axis=2)
        bounds_result = self.evaluateNumeric(bounds_numeric, options)
        
        # Convert back to interval
        bounds = Interval(bounds_result[:, :, 0], bounds_result[:, :, 1])
        return bounds
    
    def evaluatePolyZonotope(self, c: np.ndarray, G: np.ndarray, GI: np.ndarray, 
                            E: np.ndarray, id_: List[int], id_2: List[int], 
                            ind: List[int], ind_2: List[int], 
                            options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                           np.ndarray, List[int], List[int], 
                                                           List[int], List[int]]:
        """
        Evaluate polynomial zonotope (default implementation throws error)
        
        Args:
            c: Center
            G: Generators
            GI: Independent generators
            E: Exponent matrix
            id_: ID vector
            id_2: ID vector 2
            ind: Index vector
            ind_2: Index vector 2
            options: Evaluation options
            
        Returns:
            Tuple of evaluation results
            
        Raises:
            NotImplementedError: If polyZonotope evaluation is not supported
        """
        raise NotImplementedError(f"PolyZonotope evaluation not supported for {self.__class__.__name__}")
    
    def evaluateZonotopeBatch(self, c: np.ndarray, G: np.ndarray, 
                             options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate zonotope batch (default implementation throws error)
        
        Args:
            c: Center
            G: Generators
            options: Evaluation options
            
        Returns:
            Tuple of (c, G) results
            
        Raises:
            NotImplementedError: If zonotope batch evaluation is not supported
        """
        raise NotImplementedError(f"Zonotope batch evaluation not supported for {self.__class__.__name__}")
    
    def evaluateTaylm(self, input_data: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate Taylor model (default implementation throws error)
        
        Args:
            input_data: Input data
            options: Evaluation options
            
        Returns:
            r: Taylor model evaluation result
            
        Raises:
            NotImplementedError: If Taylor model evaluation is not supported
        """
        raise NotImplementedError(f"Taylor model evaluation not supported for {self.__class__.__name__}")
    
    def evaluateConZonotope(self, c: np.ndarray, G: np.ndarray, C: np.ndarray, 
                           d: np.ndarray, l: np.ndarray, u: np.ndarray, 
                           options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray]:
        """
        Evaluate constraint zonotope (default implementation throws error)
        
        Args:
            c: Center
            G: Generators
            C: Constraint matrix
            d: Constraint vector
            l: Lower bounds
            u: Upper bounds
            options: Evaluation options
            
        Returns:
            Tuple of evaluation results
            
        Raises:
            NotImplementedError: If constraint zonotope evaluation is not supported
        """
        raise NotImplementedError(f"Constraint zonotope evaluation not supported for {self.__class__.__name__}")
    
    def backpropNumeric(self, input_data: np.ndarray, grad_out: np.ndarray, 
                        options: Dict[str, Any]) -> np.ndarray:
        """
        Backpropagate numeric gradients (default implementation throws error)
        
        Args:
            input_data: Input data
            grad_out: Output gradients
            options: Backpropagation options
            
        Returns:
            grad_in: Input gradients
            
        Raises:
            NotImplementedError: If numeric backpropagation is not supported
        """
        raise NotImplementedError(f"Numeric backpropagation not supported for {self.__class__.__name__}")
    
    def backpropIntervalBatch(self, l: np.ndarray, u: np.ndarray, gl: np.ndarray, 
                             gu: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backpropagate interval batch (default implementation throws error)
        
        Args:
            l: Lower bounds
            u: Upper bounds
            gl: Lower bound gradients
            gu: Upper bound gradients
            options: Backpropagation options
            
        Returns:
            Tuple of (l, u) results
            
        Raises:
            NotImplementedError: If interval batch backpropagation is not supported
        """
        raise NotImplementedError(f"Interval batch backpropagation not supported for {self.__class__.__name__}")
    
    def backpropZonotopeBatch(self, c: np.ndarray, G: np.ndarray, gc: np.ndarray, 
                              gG: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backpropagate zonotope batch (default implementation throws error)
        
        Args:
            c: Center
            G: Generators
            gc: Center gradients
            gG: Generator gradients
            options: Backpropagation options
            
        Returns:
            Tuple of (c, G) results
            
        Raises:
            NotImplementedError: If zonotope batch backpropagation is not supported
        """
        raise NotImplementedError(f"Zonotope batch backpropagation not supported for {self.__class__.__name__}")
    
    def exportAsStruct(self) -> Dict[str, Any]:
        """
        Export layer as structure
        
        Returns:
            layerStruct: Layer structure
        """
        layerStruct = {
            'type': self.__class__.__name__,
            'name': self.name,
            'fields': self.getFieldStruct()
        }
        return layerStruct
    
    def getFieldStruct(self) -> Dict[str, Any]:
        """
        Get field structure (default implementation throws error)
        
        Returns:
            fieldStruct: Field structure
            
        Raises:
            NotImplementedError: If field struct is not supported
        """
        raise NotImplementedError(f"Field struct not supported for {self.__class__.__name__}")
    
    @classmethod
    def importFromStruct(cls, layerStruct: Dict[str, Any]) -> 'nnLayer':
        """
        Import layer from structure
        
        Args:
            layerStruct: Layer structure
            
        Returns:
            layer: Imported layer
        """
        fieldStruct = layerStruct['fields']
        
        # init layer
        layer_type = layerStruct['type']
        
        if layer_type == 'nnLinearLayer':
            W = fieldStruct['W']
            W = W.reshape(fieldStruct['size_W'][0], -1)
            b = fieldStruct['b']
            b = b.reshape(fieldStruct['size_W'][0], -1)
            
            # Import the actual class when available
            from .linear.nnLinearLayer import nnLinearLayer
            layer = nnLinearLayer(W, b)
            
            d = fieldStruct.get('d')
            if d is not None:
                d = Interval(d['inf'], d['sup'])
                d = d.reshape(fieldStruct['size_W'][0], -1)
                layer.d = d
                
        elif layer_type == 'nnElementwiseAffineLayer':
            scale = fieldStruct['scale']
            offset = fieldStruct['offset']
            from .linear.nnElementwiseAffineLayer import nnElementwiseAffineLayer
            layer = nnElementwiseAffineLayer(scale, offset)
            
        elif layer_type == 'nnReLULayer':
            from .nonlinear.nnReLULayer import nnReLULayer
            layer = nnReLULayer()
            
        elif layer_type == 'nnLeakyReLULayer':
            alpha = fieldStruct['alpha']
            from .nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
            layer = nnLeakyReLULayer(alpha)
            
        elif layer_type == 'nnSigmoidLayer':
            from .nonlinear.nnSigmoidLayer import nnSigmoidLayer
            layer = nnSigmoidLayer()
            
        elif layer_type == 'nnTanhLayer':
            from .nonlinear.nnTanhLayer import nnTanhLayer
            layer = nnTanhLayer()
            
        elif layer_type == 'nnReshapeLayer':
            idx_out = fieldStruct['idx_out']
            from .other.nnReshapeLayer import nnReshapeLayer
            layer = nnReshapeLayer(idx_out)
            
        else:
            raise ValueError(f'Unknown layer type: {layer_type}')
        
        # update name            
        layer.name = layerStruct['name']
        
        # network reduction: read merged neurons if available
        if 'merged_neurons' in fieldStruct:
            layer.merged_neurons = fieldStruct['merged_neurons']
        
        return layer
    
    @abstractmethod
    def getNumNeurons(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Abstract method to get number of input and output neurons
        
        Returns:
            Tuple of (nin, nout) where each can be None
        """
        pass
    
    @abstractmethod
    def getOutputSize(self, inputSize: List[int]) -> List[int]:
        """
        Abstract method to get output size given input size
        
        Args:
            inputSize: Input dimensions
            
        Returns:
            outputSize: Output dimensions
        """
        pass
    
    def checkInputSize(self):
        """
        Check if input size is set
        
        Raises:
            ValueError: If input size is not set
        """
        if not self.inputSize:
            raise ValueError("Input size has to be set to compute CNNs. See neuralNetwork/setInputSize.")
    
    def updateGrad(self, name: str, grad_i: np.ndarray, options: Dict[str, Any]):
        """
        Add gradient
        
        Args:
            name: Gradient name
            grad_i: Gradient to add
            options: Options containing updateGradient flag
        """
        if 'updateGradient' not in options.get('nn', {}).get('train', {}):
            if name not in self.backprop.get('grad', {}):
                self.backprop['grad'] = {}
            if name not in self.backprop['grad']:
                self.backprop['grad'][name] = 0
            self.backprop['grad'][name] = self.backprop['grad'][name] + grad_i
        else:
            if options['nn']['train']['updateGradient']:
                if name not in self.backprop.get('grad', {}):
                    self.backprop['grad'] = {}
                if name not in self.backprop['grad']:
                    self.backprop['grad'][name] = 0
                self.backprop['grad'][name] = self.backprop['grad'][name] + grad_i
    
    def getLearnableParamNames(self) -> List[str]:
        """
        Get list of learnable properties
        
        Returns:
            names: List of learnable parameter names
        """
        return []  # default none
    
    @classmethod
    def _getCount(cls) -> int:
        """
        Get unique count for naming
        
        Returns:
            count: Unique count
        """
        cls._count += 1
        return cls._count
    
    def copy(self) -> 'nnLayer':
        """
        Create a copy of the layer
        
        Returns:
            cp: Copied layer
        """
        cp = copy.deepcopy(self)
        cp.name = f"{self.name}_copy"
        return cp
    
    def __len__(self) -> int:
        """
        Return the number of layers (always 1 for single layer)
        
        Returns:
            length: Always 1
        """
        return 1
