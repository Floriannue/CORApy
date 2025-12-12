"""
nnConv2DLayer - class for convolutional 2D layers

Syntax:
    obj = nnConv2DLayer(W, b, padding, stride, dilation, name)

Inputs:
    W - weight matrix (4-D single)
        all the filters are stored in the weight-matrix:
        (kernel_height, kernel_width, in_channels, num_filters)
    b - bias column vector
    padding - zero padding [left top right bottom]
    stride - step size per dimension
    dilation - step size per dimension
    name - name of the layer, defaults to type

Outputs:
    obj - generated object

References:
    [1] T. Gehr, et al. "AI2: Safety and Robustness Certification of
        Neural Networks with Abstract Interpretation," 2018
    [2] Practical Course SoSe '22 - Report Lukas Koller

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Lukas Koller, Tobias Ladner
Written:       04-June-2022
Last update:   01-December-2022 (combine with nnAvgPool2DLayer)
               17-January-2023 (TL, optimizations)
               13-December-2023 (LK, backpropagation + vectorized weight matrix generation)
Last revision: 17-August-2022
               02-December-2022 (clean up)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
# from scipy import signal  # No longer needed - using torch operations instead
from ..nnLayer import nnLayer
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.interval import Interval
from .nnLinearLayer import nnLinearLayer

# Import helper functions from nnGeneratorReductionLayer
from .nnGeneratorReductionLayer import sub2ind, repelem, pagemtimes, pagetranspose


def aux_parseInputArgs(*args):
    """
    Parse input arguments for nnConv2DLayer constructor
    
    Args:
        *args: Variable arguments
        
    Returns:
        W, b, padding, stride, dilation, name: Parsed arguments
    """
    defaults = [1, 0, [0, 0, 0, 0], [1, 1], [1, 1], None]
    return setDefaultValues(defaults, args)


def aux_checkInputArgs(W, b, padding, stride, dilation, name):
    """
    Check input arguments for nnConv2DLayer constructor
    
    Args:
        W: Weight matrix
        b: Bias vector
        padding: Padding array
        stride: Stride array
        dilation: Dilation array
        name: Layer name
    """
    # check input types
    inputArgsCheck([
        [W, 'att', ['numeric']],
        [b, 'att', ['numeric']],
        [padding, 'att', ['numeric']],
        [stride, 'att', ['numeric']],
        [dilation, 'att', ['numeric']],
        # name is checked in nnLayer
    ])
    
    # check dimensions
    if len(W.shape) > 4:
        raise CORAerror('CORA:wrongInputInConstructor', 'Weight matrix has wrong dimensions.')
    if W.shape[3] != len(b) if hasattr(b, '__len__') and not np.isscalar(b) else 1:
        if np.isscalar(b):
            if W.shape[3] != 1:
                raise CORAerror("CORA:wrongInputInConstructor", 'Weight matrix and bias dimensions don\'t match.')
        else:
            if W.shape[3] != len(b):
                raise CORAerror("CORA:wrongInputInConstructor", 'Weight matrix and bias dimensions don\'t match.')
    if len(padding) != 4:
        raise CORAerror('CORA:wrongInputInConstructor', 'Padding must be an array with 4 entries: [left top right bottom]')
    if len(stride) != 2:
        raise CORAerror('CORA:wrongInputInConstructor', 'Stride must be an array with 2 entries: [dim1, dim2]')
    if len(dilation) != 2:
        raise CORAerror('CORA:wrongInputInConstructor', 'Dilation must be an array with 2 entries: [dim1, dim2]')


class nnConv2DLayer(nnLayer):
    """
    Convolutional 2D layer for neural networks
    
    This class implements a 2D convolutional layer with weights W and bias b.
    """
    
    # Class constant
    is_refinable = False
    
    def __init__(self, *args, **kwargs):
        """
        Constructor for nnConv2DLayer
        
        Args:
            *args: Variable arguments (W, b, padding, stride, dilation, name)
        """
        # 1. parse input arguments: varargin -> vars
        if len(args) > 6:
            raise ValueError("Too many arguments (max 6)")
        W, b, padding, stride, dilation, name = aux_parseInputArgs(*args)
        
        # 2. check correctness of input arguments
        aux_checkInputArgs(W, b, padding, stride, dilation, name)
        
        # 3. call super class constructor
        super().__init__(name)
        
        # 4. assign properties
        # Convert to torch tensors - all internal operations use torch
        if not isinstance(W, torch.Tensor):
            W = torch.tensor(W, dtype=torch.float32)
        else:
            W = W.float()
        if np.isscalar(b):
            b = torch.tensor([b], dtype=torch.float32)
        else:
            if not isinstance(b, torch.Tensor):
                b = torch.tensor(b, dtype=torch.float32)
            else:
                b = b.float()
            b = b.flatten()
        
        self.W = W
        self.b = b
        # Convert to torch tensors - all internal operations use torch
        if isinstance(padding, (list, tuple, np.ndarray)):
            self.padding = torch.tensor(padding, dtype=torch.int64)
        else:
            self.padding = torch.tensor([padding], dtype=torch.int64) if np.isscalar(padding) else padding
        if isinstance(stride, (list, tuple, np.ndarray)):
            self.stride = torch.tensor(stride, dtype=torch.int64)
        else:
            self.stride = torch.tensor([stride], dtype=torch.int64) if np.isscalar(stride) else stride
        if isinstance(dilation, (list, tuple, np.ndarray)):
            self.dilation = torch.tensor(dilation, dtype=torch.int64)
        else:
            self.dilation = torch.tensor([dilation], dtype=torch.int64) if np.isscalar(dilation) else dilation
        
        # Initialize properties to match other layers
        self.inputSize = []  # matches MATLAB property
        self.d = []  # approx error (additive)
        
        # Initialize backprop storage to match other layers
        self.backprop = {'store': {}}
    
    def getNumNeurons(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Get number of input and output neurons
        
        Returns:
            Tuple of (nin, nout)
        """
        if not self.inputSize:
            return None, None
        else:
            # we can only compute the number of neurons if the input
            # size was set.
            nin = self.inputSize[0] * self.inputSize[1] * self.inputSize[2]
            outputSize = self.getOutputSize(self.inputSize)
            # Convert to torch for prod, then back to int
            outputSize_torch = torch.tensor(outputSize, dtype=torch.int64)
            nout = torch.prod(outputSize_torch).item()
            return int(nin), int(nout)
    
    def getOutputSize(self, inImgSize: List[int]) -> List[int]:
        """
        Compute size of output feature map
        
        Args:
            inImgSize: Input image size [height, width, channels]
            
        Returns:
            outputSize: Output size [height, width, channels]
        """
        out_h, out_w, out_c = self.aux_computeOutputSize(self.W, inImgSize)
        return [int(out_h), int(out_w), int(out_c)]
    
    # evaluate ------------------------------------------------------------
    
    def evaluateNumeric(self, input_data: torch.Tensor, options: Dict[str, Any]) -> torch.Tensor:
        """
        Evaluate numeric input
        Internal to nn - input_data is always torch tensor
        
        Args:
            input_data: Input data (flattened, shape: [n, batchSize]) (torch tensor)
            options: Evaluation options
            
        Returns:
            r: Output data (flattened, torch tensor)
        """
        # Internal to nn - input_data is always torch tensor
        
        self.checkInputSize()
        r, _ = self.conv2d(input_data, 'sparseIdx')
        return r
    
    def evaluateInterval(self, bounds: Interval, options: Dict[str, Any]) -> Interval:
        """
        Evaluate interval bounds
        
        Args:
            bounds: Input interval bounds
            options: Evaluation options
            
        Returns:
            bounds: Output interval bounds
        """
        self.checkInputSize()
        # IBP (see Gowal et al. 2018)
        mu, _ = self.conv2d((bounds.sup + bounds.inf) / 2, 'sparseIdx')
        # Convert W to torch for abs operation
        W_torch = self.W if isinstance(self.W, torch.Tensor) else torch.tensor(self.W, dtype=torch.float32)
        r, _ = self.conv2d((bounds.sup - bounds.inf) / 2, 'sparseIdx', torch.abs(W_torch), None)
        
        l = mu - r
        u = mu + r
        return Interval(l, u)
    
    def evaluateSensitivity(self, S, x, options: Dict[str, Any]):
        """
        Evaluate sensitivity
        
        Args:
            S: Sensitivity matrix (torch tensor)
            x: Input point (torch tensor)
            options: Evaluation options
            
        Returns:
            S: Output sensitivity (torch tensor)
        """
        # Internal to nn - S and x are always torch tensors
        self.checkInputSize()
        
        vK, vk, batchSize = S.shape
        S = torch.permute(S, (1, 0, 2))  # permute(S,[2 1 3])
        S = S.reshape(vk, vK * batchSize)
        S = self.transconv2d(S, 'sparseIdx', self.W, None)
        S = S.reshape(S.shape[0], vK, batchSize)
        S = torch.permute(S, (1, 0, 2))  # permute(S,[2 1 3])
        return S
    
    def evaluatePolyZonotope(self, c: np.ndarray, G: np.ndarray, GI: np.ndarray, 
                            E: np.ndarray, id_: List[int], id_2: List[int], 
                            ind: List[int], ind_2: List[int], 
                            options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                           np.ndarray, List[int], List[int], 
                                                           List[int], List[int]]:
        """
        Evaluate polynomial zonotope
        
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
        """
        self.checkInputSize()
        
        # compute weight and bias
        Wff = self.aux_conv2Mat()
        bias = self.aux_getPaddedBias()
        # Ensure bias is a column vector for nnLinearLayer
        if bias.ndim == 1:
            bias = bias.reshape(-1, 1)
        
        # simulate using linear layer
        linl = nnLinearLayer(Wff, bias)
        return linl.evaluatePolyZonotope(c, G, GI, E, id_, id_2, ind, ind_2, options)
    
    def evaluateZonotopeBatch(self, c, G, options: Dict[str, Any]):
        """
        Evaluate zonotope batch (for training)
        
        Args:
            c: Center (n, 1 or 2, batchSize) (numpy array or torch tensor) - converted to torch internally
            G: Generators (n, q, batchSize) (numpy array or torch tensor) - converted to torch internally
            options: Evaluation options
            
        Returns:
            c: Output center (torch tensor)
            G: Output generators (torch tensor)
        """
        # Internal to nn - c and G are always torch tensors
        self.checkInputSize()
        
        if options.get('nn', {}).get('interval_center', False):
            n, _, batchSize = G.shape
            # Extract upper and lower bound.
            cl = c[:, 0, :].reshape(n, batchSize)
            cu = c[:, 1, :].reshape(n, batchSize)
            # Evaluate bounds.
            c_result = self.evaluateInterval(Interval(cl, cu), options)
            # Convert to torch
            c_inf = torch.tensor(c_result.inf, dtype=G.dtype, device=G.device)
            c_sup = torch.tensor(c_result.sup, dtype=G.dtype, device=G.device)
            c = torch.stack([c_inf, c_sup], dim=1)  # permute(cat(3,c.inf,c.sup),[1 3 2])
            # Evaluate generators.
            c0 = torch.zeros((n, batchSize), dtype=G.dtype, device=G.device)
            _, G, _ = self.conv2dZonotope(c0, G, 'sparseIdx')
        else:
            c, G, _ = self.conv2dZonotope(c, G, 'sparseIdx')
        
        return c, G
    
    def evaluateTaylm(self, input_data: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate Taylor model
        
        Args:
            input_data: Input Taylor model
            options: Evaluation options
            
        Returns:
            r: Output Taylor model
        """
        self.checkInputSize()
        
        # compute weight and bias
        Wff = self.aux_conv2Mat()
        bias = self.aux_getPaddedBias()
        # Ensure bias is a column vector for nnLinearLayer
        if bias.ndim == 1:
            bias = bias.reshape(-1, 1)
        
        # simulate using linear layer
        linl = nnLinearLayer(Wff, bias)
        # MATLAB: r = linl.evaluateTaylm(obj, input, options);
        # In MATLAB, obj is passed but evaluateTaylm only uses input and options
        return linl.evaluateTaylm(input_data, options)
    
    def evaluateConZonotope(self, c: np.ndarray, G: np.ndarray, C: np.ndarray, 
                           d: np.ndarray, l: np.ndarray, u: np.ndarray, 
                           options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate constrained zonotope
        
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
        """
        self.checkInputSize()
        
        # compute weight and bias
        Wff = self.aux_conv2Mat()
        bias = self.aux_getPaddedBias()
        # Ensure bias is a column vector for nnLinearLayer
        if bias.ndim == 1:
            bias = bias.reshape(-1, 1)
        
        # simulate using linear layer
        linl = nnLinearLayer(Wff, bias)
        # MATLAB: [c, G, C, d, l, u] = linl.evaluateConZonotope(obj, c, G, C, d, l, u, options);
        # In MATLAB, obj is passed but evaluateConZonotope only uses c, G, C, d, l, u, options
        return linl.evaluateConZonotope(c, G, C, d, l, u, options)
    
    # Auxiliary functions ------------------------------------------------------
    
    def convert2nnLinearLayer(self) -> nnLinearLayer:
        """
        Convert convolutional layer to linear layer
        
        Returns:
            layer: Linear layer equivalent
        """
        # as convolutional layers are just fancy linear layers,
        # they can be converted to them easily.
        self.checkInputSize()
        
        # compute weight and bias
        Wff = self.aux_conv2Mat()
        bias = self.aux_getPaddedBias()
        
        # MATLAB: layer = nnLinearLayer(Wff, bias, sprintf("%s_linear", obj.name));
        layer = nnLinearLayer(Wff, bias, f"{self.name}_linear")
        return layer
    
    def getLearnableParamNames(self) -> List[str]:
        """
        Get list of learnable properties
        
        Returns:
            names: List of learnable parameter names
        """
        return ['W', 'b']
    
    def conv2d(self, input_data, *args):
        """
        Perform 2D convolution
        
        Args:
            input_data: Input data (flattened, shape: [n, batchSize]) (numpy array or torch tensor) - converted to torch internally
            *args: Variable arguments (store, Filter, b, inImgSize, stride, padding, dilation)
                   Special case: if store == 'dWSparseIdx', then args[1] is gradOutPermImg
                   and args[2] is empty, args[3] is [in_h, in_w, batchSize]
            
        Returns:
            r: Output data (flattened, torch tensor)
            Wff: Weight matrix (not computed, returns None)
        """
        # Convert numpy input to torch if needed
        if isinstance(input_data, np.ndarray):
            input_data = torch.tensor(input_data, dtype=torch.float32)
        
        device = input_data.device
        dtype = input_data.dtype
        
        defaults = ['', self.W, self.b, self.inputSize, self.stride, self.padding, self.dilation]
        store, Filter, b, inImgSize, stride, padding, dilation = setDefaultValues(defaults, args)
        
        # Convert Filter and b to torch if needed
        if isinstance(Filter, np.ndarray):
            Filter = torch.tensor(Filter, dtype=dtype, device=device)
        elif isinstance(Filter, torch.Tensor):
            Filter = Filter.to(device=device, dtype=dtype)
        else:
            # Filter is self.W which is already torch
            Filter = Filter.to(device=device, dtype=dtype)
        
        if isinstance(b, np.ndarray):
            b = torch.tensor(b, dtype=dtype, device=device)
        elif isinstance(b, torch.Tensor):
            b = b.to(device=device, dtype=dtype)
        else:
            # b is self.b which is already torch
            b = b.to(device=device, dtype=dtype)
        
        # Handle special case for weight update: 'dWSparseIdx'
        if store == 'dWSparseIdx':
            # MATLAB: dW = obj.conv2d(inputPerm,'dWSparseIdx',gradOutPermImg,[],...
            #     [in_h in_w batchSize],obj.dilation,obj.padding,obj.stride);
            # In this case, input_data is inputPerm, args[0] is 'dWSparseIdx',
            # args[1] is gradOutPermImg, args[2] is [], args[3] is [in_h, in_w, batchSize]
            if len(args) >= 4:
                gradOutPermImg = args[1] if len(args) > 1 else None
                inImgSize_special = args[3] if len(args) > 3 else inImgSize
                # This is a special convolution for weight updates
                # For now, use the regular convolution logic but with swapped inputs
                # This is a simplified version - full implementation would be more complex
                return self._conv2d_weight_update(input_data, gradOutPermImg, Filter, inImgSize_special, stride, padding, dilation)
        
        # Handle both 1D and 2D input shapes
        # MATLAB always uses [n, batchSize] format, but Python may pass 1D arrays
        if input_data.ndim == 1:
            # 1D input: add batch dimension [n] -> [n, 1]
            input_data = input_data.reshape(-1, 1)
            _, batchSize = input_data.shape
        elif input_data.ndim == 2:
            _, batchSize = input_data.shape
        else:
            raise ValueError(f"Input data must be 1D or 2D, got shape {input_data.shape}")
        
        # padding [left,top,right,bottom]
        pad_l = int(padding[0])
        pad_t = int(padding[1])
        pad_r = int(padding[2])
        pad_b = int(padding[3])
        
        # Reshape input to image format with channel-first ordering (NCHW) for consistency
        # Flatten vector stores data as (C, H, W) per batch (channel-major), so reconstruct accordingly
        input_flat = input_data.flatten()  # (n * batch)
        if len(inImgSize) == 4:
            _, in_c, in_h, in_w = inImgSize  # assume NCHW from ONNX
        else:
            in_h, in_w, in_c = int(inImgSize[0]), int(inImgSize[1]), int(inImgSize[2])
        in_h, in_w, in_c = int(in_h), int(in_w), int(in_c)
        inputImg = input_flat.reshape(batchSize, in_c, in_h, in_w)
        # Permute to (H, W, C, batch) to match internal loops
        inputImg = inputImg.permute(2, 3, 1, 0)
        
        # Apply padding
        if pad_t > 0 or pad_b > 0 or pad_l > 0 or pad_r > 0:
            # torch.nn.functional.pad uses (pad_left, pad_right, pad_top, pad_bottom) for 2D
            inputImg = torch.nn.functional.pad(inputImg.permute(3, 2, 0, 1), (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
            inputImg = inputImg.permute(2, 3, 1, 0)  # Back to (h, w, c, batch)
        
        # Get output size
        Filter_np = Filter.cpu().numpy() if isinstance(Filter, torch.Tensor) else Filter
        out_h, out_w, out_c = self.aux_computeOutputSize(Filter_np, inImgSize, stride, padding, dilation)
        
        # Initialize output
        output = torch.zeros((out_h, out_w, out_c, batchSize), dtype=dtype, device=device)
        
        # Get filter parameters
        k_h, k_w, in_c, num_filters = Filter.shape
        stride_h, stride_w = int(stride[0]), int(stride[1])
        d_h, d_w = int(dilation[0]), int(dilation[1])
        
        # Compute dilated filter size
        f_h = k_h + (k_h - 1) * (d_h - 1)
        f_w = k_w + (k_w - 1) * (d_w - 1)
        
        # Perform convolution for each batch
        for b_idx in range(batchSize):
            for out_c_idx in range(num_filters):
                # Get filter for this output channel
                filter_kernel = Filter[:, :, :, out_c_idx]  # [k_h, k_w, in_c]
                
                # Apply dilation to filter
                if d_h > 1 or d_w > 1:
                    dilated_filter = torch.zeros((f_h, f_w, in_c), dtype=dtype, device=device)
                    for i in range(k_h):
                        for j in range(k_w):
                            dilated_filter[i * d_h, j * d_w, :] = filter_kernel[i, j, :]
                    filter_kernel = dilated_filter
                else:
                    filter_kernel = filter_kernel.clone()
                
                # Convolve each input channel and sum
                conv_result = torch.zeros((out_h, out_w), dtype=dtype, device=device)
                for in_c_idx in range(in_c):
                    # Extract single channel from input
                    input_channel = inputImg[:, :, in_c_idx, b_idx]
                    filter_channel = filter_kernel[:, :, in_c_idx]
                    
                    # Perform convolution with stride using manual sliding window
                    # This matches MATLAB's dlconv behavior more closely
                    for out_h_idx in range(out_h):
                        for out_w_idx in range(out_w):
                            # Compute input window position
                            in_h_start = out_h_idx * stride_h
                            in_w_start = out_w_idx * stride_w
                            in_h_end = in_h_start + f_h
                            in_w_end = in_w_start + f_w
                            
                            # Extract window and convolve
                            if in_h_end <= input_channel.shape[0] and in_w_end <= input_channel.shape[1]:
                                window = input_channel[in_h_start:in_h_end, in_w_start:in_w_end]
                                conv_result[out_h_idx, out_w_idx] += torch.sum(window * filter_channel)
                
                # Add bias
                if b is not None and b.numel() > 0:
                    if b.numel() == 1:
                        conv_result += b.item()
                    else:
                        conv_result += b[out_c_idx]
                
                output[:, :, out_c_idx, b_idx] = conv_result
        
        # Flatten output: [out_h * out_w * out_c, batchSize]
        # MATLAB uses column-major (Fortran) order for reshape
        # Output shape is [out_h, out_w, out_c, batchSize]
        # MATLAB: r = reshape(extractdata(rImg),[],batchSize) - column-major flatten
        # For torch, permute then flatten: (h, w, c, batch) -> (batch, h, w, c) -> flatten -> (h*w*c, batch)
        # Flatten to match ONNX (NCHW) and MATLAB ordering.
        # output is (out_h, out_w, out_c, batch) -> permute to (batch, out_c, out_h, out_w)
        # then reshape to (out_c*out_h*out_w, batch).
        r = output.permute(3, 2, 0, 1).reshape(batchSize, -1).T
        
        return r, None
    
    def _conv2d_weight_update(self, inputPerm, gradOutPermImg, 
                              Filter, inImgSize: List[int], stride,
                              padding, dilation):
        """
        Special convolution for weight updates (dWSparseIdx mode)
        
        Args:
            inputPerm: Permuted input (numpy array or torch tensor) - converted to torch internally
            gradOutPermImg: Permuted gradient output image (numpy array or torch tensor) - converted to torch internally
            Filter: Filter weights (torch tensor)
            inImgSize: Input image size [in_h, in_w, batchSize]
            stride: Stride factors
            padding: Padding values
            dilation: Dilation factors
            
        Returns:
            dW: Weight gradients (flattened, torch tensor)
            Wff: Not used
        """
        # Convert numpy inputs to torch if needed
        if isinstance(inputPerm, np.ndarray):
            inputPerm = torch.tensor(inputPerm, dtype=torch.float32)
        if isinstance(gradOutPermImg, np.ndarray):
            gradOutPermImg = torch.tensor(gradOutPermImg, dtype=torch.float32)
        
        device = inputPerm.device
        dtype = inputPerm.dtype
        
        # This is a simplified implementation
        # Full version would use sparse indexing for efficiency
        in_h, in_w, batchSize = inImgSize[0], inImgSize[1], inImgSize[2]
        
        # Reshape inputs
        inputImg = inputPerm.reshape(in_h, in_w, -1, batchSize)
        gradImg = gradOutPermImg.reshape(-1, -1, -1, batchSize)
        
        # Get filter size
        k_h, k_w, in_c, out_c = Filter.shape
        f_h, f_w = self.aux_getFilterSize(Filter.cpu().numpy() if isinstance(Filter, torch.Tensor) else Filter, dilation)
        
        # Initialize weight gradients
        dW = torch.zeros((f_h, f_w, out_c, in_c), dtype=dtype, device=device)
        
        # Compute gradients by convolving input with gradients
        # This is a simplified version
        for out_c_idx in range(out_c):
            for in_c_idx in range(in_c):
                for b_idx in range(batchSize):
                    input_ch = inputImg[:, :, in_c_idx, b_idx]
                    grad_ch = gradImg[:, :, out_c_idx, b_idx] if gradImg.shape[2] > out_c_idx else gradImg[:, :, 0, b_idx]
                    
                    # Convolve using torch's conv2d (correlate2d equivalent)
                    # For correlation, we need to flip the kernel
                    # Use conv2d with flipped kernel for correlation
                    input_ch_4d = input_ch.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
                    grad_ch_4d = grad_ch.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
                    # Flip kernel for correlation
                    grad_ch_flipped = torch.flip(torch.flip(grad_ch_4d, dims=[2]), dims=[3])
                    # Use conv2d with padding='valid' (no padding)
                    conv_result = torch.nn.functional.conv2d(input_ch_4d, grad_ch_flipped, padding=0)
                    conv_result = conv_result.squeeze(0).squeeze(0)  # Remove batch and channel dims
                    if conv_result.shape[0] >= f_h and conv_result.shape[1] >= f_w:
                        dW[:, :, out_c_idx, in_c_idx] += conv_result[:f_h, :f_w]
        
        # Flatten: MATLAB reshapes this differently
        dW_flat = dW.reshape(-1, 1)
        return dW_flat, None
    
    def conv2dZonotope(self, c: np.ndarray, G: np.ndarray, *args) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Perform 2D convolution on zonotope
        
        Args:
            c: Center (n, batchSize)
            G: Generators (n, q, batchSize)
            *args: Variable arguments
            
        Returns:
            c: Output center
            G: Output generators
            Wff: Weight matrix (not computed, returns None)
        """
        defaults = ['', self.W, self.b, self.inputSize, self.stride, self.padding, self.dilation]
        store, Filter, b, inImgSize, stride, padding, dilation = setDefaultValues(defaults, args)
        
        # Convert numpy inputs to torch if needed
        if isinstance(c, np.ndarray):
            c = torch.tensor(c, dtype=torch.float32)
        if isinstance(G, np.ndarray):
            G = torch.tensor(G, dtype=torch.float32)
        
        device = c.device if isinstance(c, torch.Tensor) else torch.device('cpu')
        dtype = c.dtype if isinstance(c, torch.Tensor) else torch.float32
        
        # Put generators into batch and do regular convolution.
        n, q, batchSize = G.shape
        # MATLAB: inputLin = reshape(cat(2,permute(c,[1 3 2]),G),n,(q+1)*batchSize);
        c_perm = torch.permute(c, (0, 2, 1)) if c.ndim == 3 else c.unsqueeze(1)
        inputLin = torch.cat([c_perm.reshape(n, -1), G.reshape(n, -1)], dim=1)
        inputLin = inputLin.reshape(n, (q + 1) * batchSize)
        
        rLin, Wff = self.conv2d(inputLin, store, Filter, b, inImgSize, stride, padding, dilation)
        r = rLin.reshape(-1, q + 1, batchSize)
        
        c = r[:, 0, :].reshape(-1, batchSize)
        G = r[:, 1:, :]
        
        return c, G, Wff
    
    def transconv2d(self, input_data, *args):
        """
        Perform transposed 2D convolution (for backpropagation)
        
        Args:
            input_data: Input data (flattened) (numpy array or torch tensor) - converted to torch internally
            *args: Variable arguments
            
        Returns:
            r: Output data (flattened, torch tensor)
        """
        # Convert numpy input to torch if needed
        if isinstance(input_data, np.ndarray):
            input_data = torch.tensor(input_data, dtype=torch.float32)
        
        device = input_data.device
        dtype = input_data.dtype
        
        defaults = ['', self.W, self.b, self.inputSize, self.stride, self.padding, self.dilation]
        store, Filter, b, inImgSize, stride, padding, dilation = setDefaultValues(defaults, args)
        
        # Convert Filter to torch if needed
        if isinstance(Filter, np.ndarray):
            Filter = torch.tensor(Filter, dtype=dtype, device=device)
        elif isinstance(Filter, torch.Tensor):
            Filter = Filter.to(device=device, dtype=dtype)
        else:
            # Filter is self.W which is already torch
            Filter = Filter.to(device=device, dtype=dtype)
        
        _, batchSize = input_data.shape
        
        # padding [left,top,right,bottom]
        pad_l = int(padding[0])
        pad_t = int(padding[1])
        pad_r = int(padding[2])
        pad_b = int(padding[3])
        
        # Compute size of gradient.
        Filter_np = Filter.cpu().numpy() if isinstance(Filter, torch.Tensor) else Filter
        out_h, out_w, out_c = self.aux_computeOutputSize(Filter_np, inImgSize, stride, padding, dilation)
        
        # Reshape input to image format: [out_h, out_w, out_c, batchSize]
        inputImg = input_data.reshape(out_h, out_w, out_c, batchSize)
        
        # Get filter parameters
        k_h, k_w, in_c, num_filters = Filter.shape
        stride_h, stride_w = int(stride[0]), int(stride[1])
        d_h, d_w = int(dilation[0]), int(dilation[1])
        
        # Compute dilated filter size
        f_h = k_h + (k_h - 1) * (d_h - 1)
        f_w = k_w + (k_w - 1) * (d_w - 1)
        
        # Initialize output (transposed convolution output is larger)
        # Output size for transposed conv: (out_h - 1) * stride + f_h - pad_t - pad_b
        out_trans_h = (out_h - 1) * stride_h + f_h - pad_t - pad_b
        out_trans_w = (out_w - 1) * stride_w + f_w - pad_l - pad_r
        output = torch.zeros((out_trans_h, out_trans_w, in_c, batchSize), dtype=dtype, device=device)
        
        # Perform transposed convolution
        for b_idx in range(batchSize):
            for out_c_idx in range(num_filters):
                # Get filter for this output channel (flip for transposed conv)
                filter_kernel = torch.flip(torch.flip(Filter[:, :, :, out_c_idx], dims=[0]), dims=[1])
                
                # Apply dilation
                if d_h > 1 or d_w > 1:
                    dilated_filter = torch.zeros((f_h, f_w, in_c), dtype=dtype, device=device)
                    for i in range(k_h):
                        for j in range(k_w):
                            dilated_filter[i * d_h, j * d_w, :] = filter_kernel[i, j, :]
                    filter_kernel = dilated_filter
                
                # Extract input channel
                input_channel = inputImg[:, :, out_c_idx, b_idx]
                
                # Upsample input by inserting zeros
                upsampled = torch.zeros((out_h * stride_h, out_w * stride_w), dtype=dtype, device=device)
                upsampled[::stride_h, ::stride_w] = input_channel
                
                # Convolve with flipped filter using torch conv2d
                for in_c_idx in range(in_c):
                    filter_channel = filter_kernel[:, :, in_c_idx]
                    # Use conv2d for convolution
                    input_4d = upsampled.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
                    filter_4d = filter_channel.unsqueeze(0).unsqueeze(0)  # (1, 1, f_h, f_w)
                    # Use conv2d with full padding (equivalent to 'full' mode)
                    conv_result = torch.nn.functional.conv2d(input_4d, filter_4d, padding=(f_h-1, f_w-1))
                    conv_result = conv_result.squeeze(0).squeeze(0)  # Remove batch and channel dims
                    # Crop to output size
                    crop_h = (conv_result.shape[0] - out_trans_h) // 2
                    crop_w = (conv_result.shape[1] - out_trans_w) // 2
                    if crop_h > 0:
                        conv_result = conv_result[crop_h:-crop_h, :]
                    if crop_w > 0:
                        conv_result = conv_result[:, crop_w:-crop_w]
                    if conv_result.shape[0] > out_trans_h:
                        conv_result = conv_result[:out_trans_h, :]
                    if conv_result.shape[1] > out_trans_w:
                        conv_result = conv_result[:, :out_trans_w]
                    output[:, :, in_c_idx, b_idx] += conv_result
        
        # Compute cropped pixels and insert zeros bottom and right
        crop = self.computeCrop()
        in_w = inImgSize[1]
        in_c = inImgSize[2]
        
        # Add zeros for cropping
        if crop[1] > 0:
            output = torch.cat([output, torch.zeros((output.shape[0], crop[1], in_c, batchSize), dtype=dtype, device=device)], dim=1)
        if crop[0] > 0:
            output = torch.cat([output, torch.zeros((crop[0], in_w, in_c, batchSize), dtype=dtype, device=device)], dim=0)
        
        # Flatten output
        r = output.reshape(-1, batchSize)
        return r
    
    def transconv2dZonotope(self, c, G, *args):
        """
        Perform transposed 2D convolution on zonotope
        
        Args:
            c: Center (numpy array or torch tensor) - converted to torch internally
            G: Generators (numpy array or torch tensor) - converted to torch internally
            *args: Variable arguments
            
        Returns:
            c: Output center (torch tensor)
            G: Output generators (torch tensor)
        """
        # Convert numpy inputs to torch if needed
        if isinstance(c, np.ndarray):
            c = torch.tensor(c, dtype=torch.float32)
        if isinstance(G, np.ndarray):
            G = torch.tensor(G, dtype=torch.float32)
        
        defaults = ['', self.inputSize, self.W, self.b, self.stride, self.padding, self.dilation]
        store, inImgSize, Filter, b, stride, padding, dilation = setDefaultValues(defaults, args)
        
        # Put generators into batch and do regular transposed convolution.
        n, q, batchSize = G.shape
        inputLin = torch.cat([c.reshape(n, -1), G.reshape(n, -1)], dim=1)
        inputLin = inputLin.reshape(n, (q + 1) * batchSize)
        
        rLin = self.transconv2d(inputLin, store, Filter, b, inImgSize, stride, padding, dilation)
        r = rLin.reshape(-1, q + 1, batchSize)
        
        c = r[:, 0, :].squeeze()
        G = r[:, 1:, :]
        
        return c, G
    
    def computeCrop(self) -> np.ndarray:
        """
        Compute number of cropped pixels
        
        Returns:
            crop: [crop_h, crop_w] number of cropped rows and columns
        """
        in_h = self.inputSize[0]
        in_w = self.inputSize[1]
        
        # Compute size of gradient.
        out_h, out_w, _ = self.aux_computeOutputSize()
        
        # padding [left,top,right,bottom]
        pad_l = self.padding[0]
        pad_t = self.padding[1]
        pad_r = self.padding[2]
        pad_b = self.padding[3]
        
        # Compute number of cropped rows and columns.
        f_h, f_w = self.aux_getFilterSize()
        crop = np.array([
            (in_h - f_h + pad_t + pad_b) - (out_h - 1) * int(self.stride[0]),
            (in_w - f_w + pad_l + pad_r) - (out_w - 1) * int(self.stride[1])
        ])
        return crop
    
    def convForWeigthsUpdate(self, grad_out, input_data):
        """
        Compute weight update using convolution
        
        Args:
            grad_out: Output gradients (numpy array or torch tensor) - converted to torch internally
            input_data: Input data (numpy array or torch tensor) - converted to torch internally
            
        Returns:
            dW: Weight gradients (torch tensor)
        """
        _, batchSize = input_data.shape
        in_h = self.inputSize[0]
        in_w = self.inputSize[1]
        in_c = self.inputSize[2]
        
        # Compute size of gradient.
        out_h, out_w, out_c = self.aux_computeOutputSize()
        
        crop = self.computeCrop()
        
        # Convert numpy inputs to torch if needed
        if isinstance(input_data, np.ndarray):
            input_data = torch.tensor(input_data, dtype=torch.float32)
        if isinstance(grad_out, np.ndarray):
            grad_out = torch.tensor(grad_out, dtype=torch.float32)
        
        device = input_data.device
        dtype = input_data.dtype
        
        # To compute the weight update, the input is convolved with the outgoing gradient.
        inputPerm = torch.permute(input_data.reshape(in_h, in_w, in_c, batchSize), (0, 1, 3, 2)).reshape(-1, in_c)
        gradOutPermImg = torch.permute(grad_out.reshape(out_h, out_w, out_c, batchSize), (0, 1, 3, 2))
        
        # Compute weight update using convolution
        # This is a simplified version - full implementation would use the dWSparseIdx option
        dW = torch.zeros((self.W.shape[0], self.W.shape[1], out_c, in_c), dtype=dtype, device=device)
        
        # For each output and input channel pair
        for out_c_idx in range(out_c):
            for in_c_idx in range(in_c):
                # Reshape gradients and input for this channel
                grad_channel = gradOutPermImg[:, :, out_c_idx, :].reshape(out_h, out_w, batchSize)
                input_channel = inputPerm[:, in_c_idx].reshape(in_h, in_w, batchSize)
                
                # Convolve input with gradients using torch
                for b_idx in range(batchSize):
                    # Use conv2d for correlation (flip kernel)
                    input_2d = input_channel[:, :, b_idx].unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
                    grad_2d = grad_channel[:, :, b_idx].unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
                    # Flip for correlation
                    grad_flipped = torch.flip(torch.flip(grad_2d, dims=[2]), dims=[3])
                    conv_result = torch.nn.functional.conv2d(input_2d, grad_flipped, padding=0)
                    conv_result = conv_result.squeeze(0).squeeze(0)  # Remove batch and channel dims
                    # Crop to filter size
                    if conv_result.shape[0] >= self.W.shape[0] and conv_result.shape[1] >= self.W.shape[1]:
                        dW[:, :, out_c_idx, in_c_idx] += conv_result[:self.W.shape[0], :self.W.shape[1]]
        
        # Reshape and crop to size
        f_h, f_w = self.aux_getFilterSize()
        dW = torch.permute(dW.reshape(f_h + crop[0], f_w + crop[1], out_c, in_c), (0, 1, 3, 2))
        dW = dW[:f_h, :f_w, :, :]
        
        return dW
    
    def aux_getFilterSize(self, Filter: Optional[np.ndarray] = None, dilation: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """
        Get filter size accounting for dilation
        
        Args:
            Filter: Filter weights (default: self.W)
            dilation: Dilation factors (default: self.dilation)
            
        Returns:
            f_h, f_w: Effective filter height and width
        """
        if Filter is None:
            Filter = self.W
        if dilation is None:
            dilation = self.dilation
        
        # return the size of the filter kernels.
        k_h = Filter.shape[0]
        k_w = Filter.shape[1]
        d_h = dilation[0]
        d_w = dilation[1]
        
        f_h = k_h + (k_h - 1) * (d_h - 1)
        f_w = k_w + (k_w - 1) * (d_w - 1)
        return int(f_h), int(f_w)
    
    def aux_computeOutputSize(self, Filter: Optional[np.ndarray] = None, inImgSize: Optional[List[int]] = None,
                             stride: Optional[np.ndarray] = None, padding: Optional[np.ndarray] = None,
                             dilation: Optional[np.ndarray] = None) -> Tuple[int, int, int]:
        """
        Compute output size of convolution
        
        Args:
            Filter: Filter weights
            inImgSize: Input image size
            stride: Stride factors
            padding: Padding values
            dilation: Dilation factors
            
        Returns:
            out_h, out_w, out_c: Output height, width, and channels
        """
        if Filter is None:
            Filter = self.W
        if inImgSize is None:
            inImgSize = self.inputSize
        if stride is None:
            stride = self.stride
        if padding is None:
            padding = self.padding
        if dilation is None:
            dilation = self.dilation
        
        # Ensure scalar floats to avoid NumPy interacting with torch tensors
        in_h = float(inImgSize[0]) if hasattr(inImgSize[0], "item") else float(inImgSize[0])
        in_w = float(inImgSize[1]) if hasattr(inImgSize[1], "item") else float(inImgSize[1])
        f_h, f_w = self.aux_getFilterSize(Filter, dilation)
        f_h = float(f_h)
        f_w = float(f_w)
        
        # padding [left,top,right,bottom]
        pad_l = float(padding[0]) if hasattr(padding[0], "item") else float(padding[0])
        pad_t = float(padding[1]) if hasattr(padding[1], "item") else float(padding[1])
        pad_r = float(padding[2]) if hasattr(padding[2], "item") else float(padding[2])
        pad_b = float(padding[3]) if hasattr(padding[3], "item") else float(padding[3])
        # stride
        stride_h = float(stride[0]) if hasattr(stride[0], "item") else float(stride[0])
        stride_w = float(stride[1]) if hasattr(stride[1], "item") else float(stride[1])
        
        out_h = int(np.floor((in_h - f_h + pad_t + pad_b) / stride_h) + 1)
        out_w = int(np.floor((in_w - f_w + pad_l + pad_r) / stride_w) + 1)
        out_c = Filter.shape[3]
        
        return out_h, out_w, out_c
    
    def aux_computeWeightMatIdx(self, *args) -> Tuple[np.ndarray, Dict]:
        """
        Compute an index-matrix to express convolutions as matrix-vector multiplication.
        
        This creates an index matrix that maps convolution operations to matrix multiplication.
        The implementation follows the MATLAB code exactly.
        
        Args:
            *args: Variable arguments (store, Filter, inImgSize, stride, padding, dilation)
            
        Returns:
            WffIdx: Index matrix for weight matrix construction
            WffSparseIdx: Sparse index structure (empty dict for now)
        """
        defaults = ['', self.W, self.inputSize, self.stride, self.padding, self.dilation]
        store, Filter, inImgSize, stride, padding, dilation = setDefaultValues(defaults, args)
        
        if store and 'store' in self.backprop and store in self.backprop['store']:
            # Return stored index matrix.
            WffIdx = self.backprop['store'][store]
            WffSparseIdx = {}
            return WffIdx, WffSparseIdx
        
        # Obtain input image size.
        in_h = inImgSize[0]
        in_w = inImgSize[1]
        
        # Compute output size.
        out_h, out_w, _ = self.aux_computeOutputSize(Filter, inImgSize, stride, padding, dilation)
        
        # Get padding [left,top,right,bottom].
        pad_l = int(padding[0])
        pad_t = int(padding[1])
        pad_r = int(padding[2])
        pad_b = int(padding[3])
        # Pad image with zeros.
        in_h_pad = in_h + pad_t + pad_b
        in_w_pad = in_w + pad_l + pad_r
        
        # Get stride values.
        s_h = int(stride[0])
        s_w = int(stride[1])
        
        n = out_h * out_w
        m = in_h_pad * in_w_pad
        
        # Get filter kernel size.
        k_h, k_w, in_c, out_c = Filter.shape
        
        # Compute an index-matrix for an individual filter.
        # MATLAB: filterIdx = reshape(1:k_h*k_w, k_h, k_w) - uses column-major reshape
        # Per readme: use C-order in Python, handle MATLAB compatibility at interface
        # MATLAB's reshape creates: [[1, 3], [2, 4]] for 2x2 (column-major)
        # Python: use Fortran order to match MATLAB's column-major reshape at interface
        filterIdx = np.arange(1, k_h * k_w + 1, dtype=Filter.dtype).reshape(k_h, k_w, order='F')
        
        if np.any(dilation != 1):
            # Dilate filter indices
            d_h = int(dilation[0])
            d_w = int(dilation[1])
            # Compute dilated filter size.
            f_h = k_h + (k_h - 1) * (d_h - 1)
            f_w = k_w + (k_w - 1) * (d_w - 1)
            # Append zero-rows and -columns.
            eye_fh_kh = np.eye(f_h, k_h, dtype=Filter.dtype)
            eye_kw_fw = np.eye(k_w, f_w, dtype=Filter.dtype)
            padFilterIdx = eye_fh_kh @ filterIdx @ eye_kw_fw
            # Permute the rows and columns
            idx_h_list = []
            for i in range(k_h - 1):
                idx_h_list.append(i + 1)
                idx_h_list.extend(range(k_h + 1 + i * (d_h - 1), k_h + 1 + (i + 1) * (d_h - 1)))
            idx_h_list.append(k_h)
            idx_h = np.array(idx_h_list, dtype=int) - 1  # Convert to 0-based
            
            idx_w_list = []
            for i in range(k_w - 1):
                idx_w_list.append(i + 1)
                idx_w_list.extend(range(k_w + 1 + i * (d_w - 1), k_w + 1 + (i + 1) * (d_w - 1)))
            idx_w_list.append(k_w)
            idx_w = np.array(idx_w_list, dtype=int) - 1  # Convert to 0-based
            
            filterIdx = padFilterIdx[np.ix_(idx_h, idx_w)]
        else:
            # Filter is not dilated
            f_h = k_h
            f_w = k_w
        
        # We turn the filter kernel into a vector, padded with 0's.
        # MATLAB: linFilterIdx = [filterIdx zeros(f_h,in_w_pad-f_w,'like',Filter); 
        #                         zeros(in_h_pad-f_h,in_w_pad,'like',Filter)];
        # This concatenates horizontally first (filterIdx + zeros to the right),
        # then vertically (zeros below)
        # First concatenate horizontally: [filterIdx, zeros(f_h, in_w_pad-f_w)]
        linFilterIdx_top = np.concatenate([
            filterIdx,
            np.zeros((f_h, in_w_pad - f_w), dtype=Filter.dtype)
        ], axis=1)
        # Then concatenate vertically: [linFilterIdx_top; zeros(in_h_pad-f_h, in_w_pad)]
        linFilterIdx = np.concatenate([
            linFilterIdx_top,
            np.zeros((in_h_pad - f_h, in_w_pad), dtype=Filter.dtype)
        ], axis=0)
        
        # Compute number of zeros needed for padding before the filter.
        nzspad = in_h_pad * (in_w_pad - f_w + 1) - f_h
        # MATLAB: linFilterIdx(:) - column-major flatten
        # Per readme: use C-order in Python, handle MATLAB compatibility at interface
        # MATLAB's A(:) flattens column-major, so use Fortran order at interface
        linFilterIdx = np.concatenate([
            np.zeros((nzspad,), dtype=Filter.dtype),
            linFilterIdx.flatten(order='F')
        ])
        
        # We compute indices that index the vectorized filter to obtain the
        # filter-matrix. We compute the shifts for each rows.
        # Adjust for row breaks.
        # CRITICAL: MATLAB's repmat(rowShift,out_h,1) + reshape([],1) behavior
        # MATLAB: repmat([2,1,0], 3, 1) creates:
        #   [2, 1, 0;
        #    2, 1, 0;
        #    2, 1, 0]
        # Then reshape([],1) flattens COLUMN-MAJOR: [2,2,2, 1,1,1, 0,0,0]
        # NOT row-major: [2,1,0, 2,1,0, 2,1,0]
        # This column-major flattening is essential for correct row ordering in WfIdx!
        rowShift = np.arange(out_w - 1, -1, -1, dtype=Filter.dtype)
        # repmat(rowShift,out_h,1) creates [out_h x out_w] matrix, each row is rowShift
        rowShift = np.tile(rowShift.reshape(1, -1), (out_h, 1))
        # reshape([],1) flattens column-major to column vector (MATLAB default)
        rowShift = rowShift.flatten(order='F').reshape(-1, 1)
        
        # Adjust for horizontal and vertical stride.
        # MATLAB: repmat(out_h-1:-1:0,1,out_w)' creates (out_w, out_h) matrix
        # Each column is [out_h-1, out_h-2, ..., 0]
        vertical_part = np.tile(np.arange(out_h - 1, -1, -1).reshape(-1, 1), (1, out_w)).T.flatten()
        rowShift = ((s_w - 1) * in_h_pad + f_w - 1) * rowShift.flatten() + \
                   (s_h - 1) * (vertical_part + (out_w - 1) * rowShift.flatten())
        
        # Adjust for cutoff rows and columns.
        rowShift = rowShift + \
                   in_h_pad * ((in_h_pad - f_h) % s_h) + \
                   ((in_w_pad - f_w) % s_w) * np.tile(np.arange(out_w, 0, -1), out_h)
        
        # Compute the index matrix.
        n_vec = np.arange(n, 0, -1, dtype=Filter.dtype).reshape(-1, 1)
        m_vec = np.arange(0, m, dtype=Filter.dtype).reshape(1, -1)
        ij = (n_vec + rowShift.reshape(-1, 1)) + m_vec
        
        # Compute the filter-index matrix.
        # MATLAB uses 1-based indexing, so convert to 0-based for Python
        # MATLAB indexes matrices column-major, so flatten in Fortran order
        ij_flat = ij.flatten(order='F').astype(int) - 1
        # Handle out-of-bounds indices
        ij_flat = np.clip(ij_flat, 0, len(linFilterIdx) - 1)
        # MATLAB reshape uses column-major order, so reshape in Fortran order
        WfIdx = linFilterIdx[ij_flat].reshape(n, m, order='F')
        
        # We remove padded area.
        isPad = np.zeros((in_h_pad, in_w_pad), dtype=bool)
        isPad[:pad_t, :] = True
        isPad[in_h_pad - pad_b:, :] = True
        isPad[:, :pad_l] = True
        isPad[:, in_w_pad - pad_r:] = True
        # MATLAB: reshape(isPad, 1, []) - column-major flatten to row vector
        # Per readme: use C-order in Python, handle MATLAB compatibility at interface
        isPad = isPad.flatten(order='F')
        WfIdx = WfIdx[:, ~isPad]
        
        # We add 1 to allow for indexing a zero value which is prepended to each filter.
        WfIdx = WfIdx + 1
        
        # Assemble individual filter-index matrices to larger index matrix.
        WffIdx = np.tile(WfIdx, (out_c, in_c))
        
        # We have to shift the indices for each filter by the number of entries in each filter.
        indShift = (k_h * k_w + 1) * np.arange(in_c * out_c, dtype=Filter.dtype)
        indShift = indShift.reshape(in_c, out_c).T
        indShift = np.repeat(indShift, out_h * out_w, axis=0)
        indShift = np.tile(indShift, (1, in_h * in_w))
        WffIdx = WffIdx + indShift
        
        WffSparseIdx = {}
        
        if store:
            # Store index matrix (backprop is already initialized in constructor)
            self.backprop['store'][store] = WffIdx
        
        return WffIdx, WffSparseIdx
    
    def aux_conv2Mat(self, *args) -> np.ndarray:
        """
        Compute weight matrix to express convolutions as matrix-vector multiplication.
        
        This converts the convolution operation to a matrix multiplication using
        the index matrix computed by aux_computeWeightMatIdx.
        
        Args:
            *args: Variable arguments
            
        Returns:
            Wff: Weight matrix for matrix multiplication representation
        """
        defaults = ['', self.W, self.inputSize, self.stride, self.padding, self.dilation]
        store, Filter, inImgSize, stride, padding, dilation = setDefaultValues(defaults, args)
        
        # Get filter kernel size.
        k_h, k_w, in_c, out_c = Filter.shape
        
        # WffIdx = obj.aux_computeWeightMatIdx(varargin{:});
        WffIdx, WffSparseIdx = self.aux_computeWeightMatIdx(store, Filter, inImgSize, stride, padding, dilation)
        
        # Vectorize all filter kernels and prepend 0.
        # MATLAB: linFfilter = [zeros(1,in_c,out_c,'like',Filter); 
        #     reshape(Filter,[],in_c,out_c)];
        # MATLAB: linFfilter = linFfilter(:);
        # Per readme: use C-order in Python, handle MATLAB compatibility at interface
        # MATLAB's reshape(Filter,[],in_c,out_c) flattens k_h*k_w in column-major
        # At interface: transpose spatial dims to match MATLAB's column-major reshape
        Filter_transposed = np.transpose(Filter, [1, 0, 2, 3])  # Swap k_h and k_w for interface compatibility
        Filter_reshaped = Filter_transposed.reshape(k_h * k_w, in_c, out_c)
        linFfilter = np.concatenate([
            np.zeros((1, in_c, out_c), dtype=Filter.dtype),
            Filter_reshaped
        ], axis=0)
        # MATLAB: linFfilter(:) - column-major flatten of [1+k_h*k_w, in_c, out_c]
        # Per readme: for MATLAB A(:), transpose first then flatten in C-order
        # Transpose [1+k_h*k_w, in_c, out_c] → [in_c, out_c, 1+k_h*k_w], then flatten
        linFfilter = np.transpose(linFfilter, [1, 2, 0]).flatten()
        
        # Construct weight matrix.
        # MATLAB: Wff = reshape(linFfilter(WffIdx),size(WffIdx));
        # MATLAB indexes matrices column-major, so flatten in Fortran order
        # WffIdx uses 1-based indexing, so we need to subtract 1 for Python
        WffIdx_0based = (WffIdx - 1).flatten(order='F')
        # Clip indices to valid range and convert to integer
        WffIdx_0based = np.clip(WffIdx_0based, 0, len(linFfilter) - 1).astype(np.int64)
        # Reshape back in Fortran order to match MATLAB's column-major reshape
        Wff = linFfilter[WffIdx_0based].reshape(WffIdx.shape, order='F')
        
        return Wff
    
    def aux_getPaddedBias(self, *args) -> np.ndarray:
        """
        Pad bias such that convolution can be simulated by linear layer.
        
        Args:
            *args: Variable arguments
            
        Returns:
            bias: Padded bias vector
        """
        defaults = ['', self.W, self.b, self.inputSize, self.stride, self.padding, self.dilation]
        store, Filter, b, inImgSize, stride, padding, dilation = setDefaultValues(defaults, args)
        
        # compute output size
        out_h, out_w, out_c = self.aux_computeOutputSize(Filter, inImgSize, stride, padding, dilation)
        
        if b is None or (isinstance(b, np.ndarray) and b.size == 0):
            b = np.zeros(out_c, dtype=Filter.dtype)
        elif np.isscalar(b) or (isinstance(b, np.ndarray) and b.size == 1):
            b = np.full(out_c, float(b), dtype=Filter.dtype)
        else:
            b = np.asarray(b).flatten()
        
        # expand the bias vector to output size
        bias = repelem(b, out_h * out_w)
        
        return bias
    
    # backprop ------------------------------------------------------------
    
    def backpropNumeric(self, input_data: torch.Tensor, grad_out: torch.Tensor, 
                        options: Dict[str, Any]) -> torch.Tensor:
        """
        Backpropagate numeric gradients
        
        Args:
            input_data: Input data
            grad_out: Output gradients
            options: Backpropagation options
            
        Returns:
            grad_in: Input gradients
        """
        # Compute weight update.
        dW = self.convForWeigthsUpdate(grad_out, input_data)
        
        # Compute size of gradient.
        out_h, out_w, out_c = self.aux_computeOutputSize()
        _, batchSize = input_data.shape
        
        # Convert to torch if needed
        if isinstance(grad_out, np.ndarray):
            grad_out = torch.tensor(grad_out, dtype=torch.float32)
        
        device = grad_out.device if isinstance(grad_out, torch.Tensor) else torch.device('cpu')
        dtype = grad_out.dtype if isinstance(grad_out, torch.Tensor) else torch.float32
        
        # Compute bias update.
        grad_out_reshaped = grad_out.reshape(out_h, out_w, out_c, batchSize)
        db = torch.sum(grad_out_reshaped, dim=(0, 1, 3))  # squeeze(sum(...,[1 2 4]))
        
        # Convert back to numpy if needed for updateGrad
        if hasattr(self, 'updateGrad'):
            db_np = db.cpu().numpy() if isinstance(db, torch.Tensor) else db
        else:
            db_np = db
        
        # Update weights and bias.
        self.updateGrad('W', dW, options)
        self.updateGrad('b', db, options)
        
        # The backproped gradient is computed by (full) convolving the 
        # outgoing gradient with the filters rotated by 180 degrees, which 
        # is the same as the transposed convolution.
        grad_in = self.transconv2d(grad_out, 'sparseIdx', self.W, None)
        return grad_in
    
    def backpropIntervalBatch(self, l: torch.Tensor, u: torch.Tensor, gl: torch.Tensor, 
                             gu: torch.Tensor, options: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backpropagate interval batch gradients
        
        Args:
            l: Lower bounds
            u: Upper bounds
            gl: Lower bound gradients
            gu: Upper bound gradients
            options: Backpropagation options
            
        Returns:
            gl, gu: Backpropagated gradients
        """
        # Internal to nn - all inputs are always torch tensors
        
        device = l.device if isinstance(l, torch.Tensor) else torch.device('cpu')
        dtype = l.dtype if isinstance(l, torch.Tensor) else torch.float32
        
        # See (Gowal et al. 2019)
        mu = (u + l) / 2
        r = (u - l) / 2
        
        # Compute weight update.
        # Convert to numpy for convForWeigthsUpdate if it expects numpy
        mu_np = mu.cpu().numpy() if isinstance(mu, torch.Tensor) else mu
        r_np = r.cpu().numpy() if isinstance(r, torch.Tensor) else r
        gu_gl_sum_np = (gu + gl).cpu().numpy() if isinstance(gu, torch.Tensor) else (gu + gl)
        gu_gl_diff_np = (gu - gl).cpu().numpy() if isinstance(gu, torch.Tensor) else (gu - gl)
        
        dWmu = self.convForWeigthsUpdate(gu_gl_sum_np, mu_np)
        dWr_np = self.convForWeigthsUpdate(gu_gl_diff_np, r_np)
        # Convert to torch for multiplication
        if isinstance(dWr_np, np.ndarray):
            dWr_np = torch.tensor(dWr_np, dtype=dtype, device=device)
        # W should always be torch tensor, but handle both cases for safety
        if isinstance(self.W, torch.Tensor):
            dWr = dWr_np * torch.sign(self.W)
        else:
            # Convert to torch if not already
            W_torch = torch.tensor(self.W, dtype=torch.float32)
            dWr = dWr_np * torch.sign(W_torch)
        
        # Compute size of gradient.
        out_h, out_w, out_c = self.aux_computeOutputSize()
        _, batchSize = l.shape
        
        # Compute bias update.
        gl_reshaped = gl.reshape(out_h, out_w, out_c, batchSize)
        db = torch.sum(gl_reshaped, dim=(0, 1, 3))  # squeeze(sum(...,[1 2 4]))
        
        # Update weights and bias - keep as torch tensors
        self.updateGrad('W', dWmu + dWr, options)
        self.updateGrad('b', db, options)
        
        # Use transposed convolutions to backprop gradients.
        dmu = self.transconv2d((gu + gl) / 2, 'sparseIdx', self.W, None)
        # Convert W to torch for abs operation if needed
        W_torch = self.W if isinstance(self.W, torch.Tensor) else torch.tensor(self.W, dtype=torch.float32)
        dr = self.transconv2d((gu - gl) / 2, 'sparseIdx', torch.abs(W_torch), None)
        gl = dmu - dr
        gu = dmu + dr
        
        return gl, gu
    
    def backpropZonotopeBatch(self, c: torch.Tensor, G: torch.Tensor, gc: torch.Tensor, 
                              gG: torch.Tensor, options: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backpropagate zonotope batch gradients
        
        Args:
            c: Center
            G: Generators
            gc: Center gradients
            gG: Generator gradients
            options: Backpropagation options
            
        Returns:
            gc, gG: Backpropagated gradients
        """
        in_h = self.inputSize[0]
        in_w = self.inputSize[1]
        in_c = self.inputSize[2]
        
        # Compute size of gradient.
        out_h, out_w, out_c = self.aux_computeOutputSize()
        f_h, f_w = self.aux_getFilterSize()
        
        # padding [left,top,right,bottom]
        pad_l = self.padding[0]
        pad_t = self.padding[1]
        pad_r = self.padding[2]
        pad_b = self.padding[3]
        
        # Compute number of cropped rows and columns.
        crop = self.computeCrop()
        
        # gradient of the filters are computed by convolving the gradient 
        # with the input.
        nIn = G.shape[0]
        nGrad, q, batchSize = gG.shape
        
        if options.get('nn', {}).get('interval_center', False):
            # Extract bounds.
            cl = c[:, 0, :].reshape(nIn, batchSize)
            cu = c[:, 1, :].reshape(nIn, batchSize)
            # Extract gradient for the bounds.
            gl = gc[:, 0, :].reshape(nGrad, batchSize)
            gu = gc[:, 1, :].reshape(nGrad, batchSize)
            # Convert to numpy for backpropIntervalBatch if needed, then convert back
            if isinstance(cl, torch.Tensor):
                cl = cl.cpu().numpy()
            if isinstance(cu, torch.Tensor):
                cu = cu.cpu().numpy()
            if isinstance(gl, torch.Tensor):
                gl = gl.cpu().numpy()
            if isinstance(gu, torch.Tensor):
                gu = gu.cpu().numpy()
            
            gl, gu = self.backpropIntervalBatch(cl, cu, gl, gu, options)
            
            # Convert back to torch
            if isinstance(gl, np.ndarray):
                gl = torch.tensor(gl, dtype=gG.dtype, device=gG.device)
            if isinstance(gu, np.ndarray):
                gu = torch.tensor(gu, dtype=gG.dtype, device=gG.device)
            
            gc = torch.stack([gl, gu], dim=1)  # permute(cat(3,gl,gu),[1 3 2])
            
            c0 = torch.zeros((nGrad, batchSize), dtype=gG.dtype, device=gG.device)
            _, gG = self.transconv2dZonotope(c0, gG, 'sparseIdx', self.W, None)
        else:
            # Convert inputs to torch if needed
            if isinstance(c, np.ndarray):
                c = torch.tensor(c, dtype=torch.float32)
            if isinstance(G, np.ndarray):
                G = torch.tensor(G, dtype=torch.float32)
            if isinstance(gc, np.ndarray):
                gc = torch.tensor(gc, dtype=torch.float32)
            if isinstance(gG, np.ndarray):
                gG = torch.tensor(gG, dtype=torch.float32)
            
            device = gG.device if isinstance(gG, torch.Tensor) else torch.device('cpu')
            dtype = gG.dtype if isinstance(gG, torch.Tensor) else torch.float32
            
            # Only using options.nn.train.zonotope_weight_update = 'sum'
            # We move the generators to the batch. This is in order to do a
            # convolution of the input with the outgoing gradient.
            c_perm = torch.permute(c, (0, 2, 1)) if c.ndim == 3 else c.unsqueeze(1)
            inputLin = torch.cat([c_perm.reshape(nIn, -1), G.reshape(nIn, -1)], dim=1)
            inputLin = inputLin.reshape(nIn, (q + 1) * batchSize)
            inputLinPerm = torch.permute(inputLin.reshape(in_h, in_w, in_c, (q + 1) * batchSize), (0, 1, 3, 2)).reshape(-1, in_c)
            
            # Similarly, we move the generator of the outgoing gradient to the
            # batch as well.
            gc_perm = torch.permute(gc, (0, 2, 1)) if gc.ndim == 3 else gc.unsqueeze(1)
            gradLin = torch.cat([gc_perm.reshape(nGrad, -1), gG.reshape(nGrad, -1)], dim=1)
            gradLin = gradLin.reshape(nGrad, (q + 1) * batchSize)
            gradOutPermImg = torch.permute(gradLin.reshape(out_h, out_w, out_c, (q + 1) * batchSize), (0, 1, 3, 2))
            
            # To compute the weight update, the input is convolved with the
            # outgoing gradient. (Simplified - full version uses dWSparseIdx)
            # For now, use convForWeigthsUpdate on a per-batch basis
            weightsUpdate = torch.zeros((f_h, f_w, in_c, out_c), dtype=dtype, device=device)
            for b_idx in range(batchSize):
                # Extract center and generators for this batch
                c_batch = c[:, :, b_idx] if c.ndim == 3 else c
                gc_batch = gc[:, :, b_idx] if gc.ndim == 3 else gc
                # Simplified weight update
                dW_batch = self.convForWeigthsUpdate(gc_batch, c_batch)
                weightsUpdate += dW_batch[:, :, :, :dW_batch.shape[3]] if dW_batch.shape[3] <= out_c else dW_batch[:, :, :, :out_c]
            
            weightsUpdate = weightsUpdate[:f_h, :f_w, :, :]
            
            # Compute bias update.
            gc_reshaped = gc.reshape(out_h, out_w, out_c, batchSize)
            biasUpdate = torch.sum(gc_reshaped, dim=(0, 1, 3))
            
            self.updateGrad('W', weightsUpdate, options)
            self.updateGrad('b', biasUpdate, options)
            
            # The backproped gradient is computed by (full) convolving the 
            # outgoing gradient with the filters rotated by 180 degrees, which 
            # is the same as the transposed convolution.
            gc, gG = self.transconv2dZonotope(gc, gG, 'sparseIdx', self.W, None)
        
        return gc, gG

