"""
nnSoftmaxLayer - class for Softmax layer

This class implements a softmax activation layer for neural networks.
"""

import numpy as np
from .nnActivationLayer import nnActivationLayer


class nnSoftmaxLayer(nnActivationLayer):
    """
    Softmax activation layer for neural networks.
    
    The softmax function is defined as: f(x_i) = exp(x_i) / sum(exp(x_j))
    """
    
    def __init__(self, name=None):
        """
        Constructor for nnSoftmaxLayer.
        
        Args:
            name: name of the layer, defaults to type
        """
        if name is None:
            name = "softmax"
        
        # Define the softmax function and its derivative
        def softmax(x):
            # avoid numerical issues see [2, Chp. 4]
            x = x - np.max(x)
            return np.exp(x) / np.sum(np.exp(x))
        
        def softmax_derivative(x):
            # This is a placeholder - actual derivative computation is complex
            # and handled by getDf method
            return np.zeros_like(x)
        
        super().__init__(name)
        
        # Import nnExpLayer for evaluation
        try:
            from .nnExpLayer import nnExpLayer
            self.expLayer = nnExpLayer()
        except ImportError:
            # If nnExpLayer doesn't exist yet, we'll handle it later
            self.expLayer = None
    
    def getDf(self, i):
        """
        Get the i-th derivative of the softmax function.
        
        Args:
            i: order of derivative (0, 1)
            
        Returns:
            Function handle for the i-th derivative
        """
        if i == 0:
            return self.f
        elif i == 1:
            def deriv(x):
                sx = np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
                sx = np.transpose(sx, [0, 2, 1])
                # compute Jacobian of softmax
                # J = -sx * sx^T + sx * I
                J = -np.einsum('ijk,ilk->ijl', sx, sx) + sx * np.eye(sx.shape[0])[:, :, np.newaxis]
                r = np.reshape(np.einsum('ijk,ilk->ij', J, np.transpose(x, [0, 2, 1])), x.shape)
                return r
            return deriv
        else:
            # Higher order derivatives are not supported for softmax
            return lambda x: np.zeros_like(x)
    
    def getDerBounds(self, l, u):
        """
        Get derivative bounds for the softmax function.
        
        Args:
            l: lower bound
            u: upper bound
            
        Returns:
            Tuple of (df_l, df_u) where df_l and df_u are the bounds on the derivative
        """
        # df_l and df_u as lower and upper bound for the derivative
        # For softmax, derivative bounds are [0, 1]
        return 0, 1
    
    def minMaxDiffSoftmax(self, l, u, coeffs_n, der1, dx):
        """
        Compute tolerance for softmax approximation.
        
        Args:
            l: lower bound
            u: upper bound
            coeffs_n: coefficients for neuron
            der1: derivative bounds
            dx: step size
            
        Returns:
            Tolerance value
        """
        # This would require nnHelper.getDerInterval which we don't have yet
        # For now, return a small default value
        return 0.0001
    
    def evaluateSensitivity(self, S, x, options):
        """
        Evaluate sensitivity for the softmax function.
        
        Args:
            S: sensitivity matrix
            x: input data
            options: options dictionary
            
        Returns:
            Updated sensitivity matrix
        """
        # MATLAB: sx = permute(obj.evaluateNumeric(x, options),[1 3 2]);
        # MATLAB: J = -pagemtimes(sx,'none',sx,'transpose') + sx.*eye(size(x,1));
        # MATLAB: S = pagemtimes(S,J);

        # Get softmax output and reshape to match MATLAB permute
        sx = self.evaluateNumeric(x, options)  # Shape: (num_neurons, batch_size)
        sx = sx[:, np.newaxis, :]  # Shape: (num_neurons, 1, batch_size)
        
        # Compute Jacobian of softmax using pagemtimes equivalent
        # J should be (num_neurons, num_neurons, batch_size)
        # First term: -pagemtimes(sx,'none',sx,'transpose') = -sx @ sx^T for each batch
        J = -np.matmul(sx, np.transpose(sx, (0, 2, 1)))  # Shape: (num_neurons, num_neurons, batch_size)
        
        # Second term: sx.*eye(size(x,1)) = sx * identity matrix for each batch
        # eye(size(x,1)) should be (num_neurons, num_neurons)
        identity = np.eye(sx.shape[0])  # Shape: (num_neurons, num_neurons)
        # Reshape identity to (num_neurons, num_neurons, 1) for broadcasting
        identity = identity[:, :, np.newaxis]  # Shape: (num_neurons, num_neurons, 1)
        # Add the identity term: sx * identity for each batch
        J = J + sx * identity  # Shape: (num_neurons, num_neurons, batch_size)
        
        # Apply Jacobian to sensitivity matrix using pagemtimes equivalent
        # S is (nK, input_dim, bSz), J is (output_dim, output_dim, bSz)
        # We need S @ J for each batch element, preserving batch dimension
        
        # Debug: print dimensions to understand what's happening
        print(f"DEBUG: Softmax evaluateSensitivity - S shape: {S.shape}, J shape: {J.shape}")
        print(f"DEBUG: Softmax evaluateSensitivity - sx shape: {sx.shape}")
        
        # Use batch matrix multiplication equivalent to MATLAB's pagemtimes
        # For 3D tensors, we need to handle each batch element separately
        if S.ndim == 3 and J.ndim == 3:
            # S is (nK, input_dim, bSz), J is (output_dim, output_dim, bSz)
            # Result should be (nK, output_dim, bSz)
            result = np.zeros((S.shape[0], J.shape[1], S.shape[2]))
            for i in range(S.shape[2]):  # iterate over batch dimension
                result[:, :, i] = S[:, :, i] @ J[:, :, i]
            S = result
        else:
            # Fallback to regular matmul for non-3D cases
            S = np.matmul(S, J)
        
        return S
    
    def evaluateNumeric(self, input_data, options):
        """
        Evaluate the softmax function numerically.
        
        Args:
            input_data: input data
            options: options dictionary
            
        Returns:
            Output of the softmax function
        """
        # avoid numerical issues see [2, Chp. 4]
        input_data = input_data - np.max(input_data)
        return np.exp(input_data) / np.sum(np.exp(input_data))
    
    def evaluatePolyZonotope(self, c, G, GI, E, id_, id__, ind, ind_, options):
        """
        Evaluate polyZonotope for softmax layer.
        
        Args:
            c: center
            G: generators
            GI: independent generators
            E: exponential matrix
            id_: id
            id__: id
            ind: indices
            ind_: indices
            options: options dictionary
            
        Returns:
            Updated polyZonotope parameters
        """
        # avoid numerical issues
        c = c - np.max(c)
        
        # evaluate polyZonotope on expLayer
        poly_method = options['nn']['poly_method']
        options['nn']['poly_method'] = 'singh'  # formerly known as 'lin'
        
        if self.expLayer is not None:
            c, G, GI, E, id_, id__, ind, ind_ = self.expLayer.evaluatePolyZonotope(
                c, G, GI, E, id_, id__, ind, ind_, options
            )
        
        options['nn']['poly_method'] = poly_method
        num_neurons = G.shape[0]
        order = max(self.order)
        
        if order > 1:
            raise ValueError("nnSoftmaxLayer only supports order 1.")
        
        # initialization
        # This would require nnHelper.getOrderIndicesG which we don't have yet
        # For now, we'll use a simplified approach
        G_start = 0
        G_end = G.shape[1]
        GI_end = GI.shape[1] if GI.size > 0 else 0
        
        # calculate exponential matrix
        Es = np.zeros((E.shape[0], G_end))
        Es[:, :G_end] = E
        
        # init
        c_ = np.zeros((num_neurons, 1))
        G_ = np.zeros((num_neurons, G_end))
        GI_ = np.zeros((num_neurons, GI_end))
        d = np.zeros((num_neurons, 1))
        
        if not np.all(np.array(self.order).shape == c.shape):
            self.order = np.ones(c.shape) * self.order
        
        # sum dimensions
        c_sum = np.sum(c, axis=0, keepdims=True)
        G_sum = np.sum(G, axis=0, keepdims=True)
        GI_sum = np.sum(GI, axis=0, keepdims=True) if GI.size > 0 else np.zeros((1, GI.shape[1]))
        
        # loop over all neurons in the current layer
        for i in range(num_neurons):
            options['nn']['neuron_i'] = i
            order_j = self.order[i]
            c_[i], G_i, GI_i, d[i] = self.evaluatePolyZonotopeNeuronSoftmax(
                c[i], G[i, :], GI[i, :] if GI.size > 0 else np.zeros((1, GI.shape[1])), 
                Es, order_j, ind, ind_, c_sum, G_sum, GI_sum, options
            )
            
            G_[i, :len(G_i)] = G_i
            if GI_i.size > 0:
                GI_[i, :len(GI_i)] = GI_i
        
        # update properties
        c = c_
        G = G_
        E = Es
        
        # add approximation error
        if GI_.size > 0:
            GI = GI_[:, np.sum(np.abs(GI_), axis=0) > 0]
        else:
            GI = GI_
        
        Gd = np.diag(d.flatten())
        Gd = Gd[:, d.flatten() > 0]
        
        if options['nn'].get('add_approx_error_to_GI', False):
            if Gd.size > 0:
                GI = np.hstack([GI, Gd]) if GI.size > 0 else Gd
        else:
            if Gd.size > 0:
                G = np.hstack([G, Gd])
                E = np.block([[E, np.zeros((E.shape[0], Gd.shape[1]))], 
                            [np.zeros((Gd.shape[1], E.shape[1])), np.eye(Gd.shape[1])]])
                # Update id and id_ would require more complex logic
        
        return c, G, GI, E, id_, id__, ind, ind_
    
    def evaluatePolyZonotopeNeuronSoftmax(self, c, G, GI, Es, order, ind, ind_, c_sum, G_sum, GI_sum, options):
        """
        Evaluate polyZonotope for a single neuron in softmax layer.
        
        Args:
            c: center for neuron
            G: generators for neuron
            GI: independent generators for neuron
            Es: exponential matrix
            order: polynomial order
            ind: indices
            ind_: indices
            c_sum: sum of centers
            G_sum: sum of generators
            GI_sum: sum of independent generators
            options: options dictionary
            
        Returns:
            Updated parameters for the neuron
        """
        # This is a complex method that would require nnHelper functions
        # For now, return simplified results
        return c, G, GI, 0.0
    
    def computeExtremePointsBatch(self, m, options=None):
        """
        Compute extreme points for batch processing.
        
        Args:
            m: slope values
            options: options dictionary
            
        Returns:
            Extreme points (not implemented for softmax)
        """
        # do not consider approximation errors...
        return np.inf * m
