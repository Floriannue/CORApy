"""
refine - refine the network using the maximum error bound

Description:
    Refine the network using the maximum error bound

Syntax:
    refine(max_order, type_, method, x, verbose, force_bounds, gamma)

Inputs:
    max_order - Maximum order for refinement
    type_ - "all", "naive", "layer"- or "neuron"-wise refinement
    method - Refinement heuristic
       "approx_error", "sensitivity", "both", "random", "all", "layer_bias"
    x - Point used for sensitivity analysis
    verbose - Whether additional information should be displayed
    force_bounds - Orders at which to re-compute bounds
    gamma - Threshold for neuron-wise refinement

Outputs:
    None

Example:
    nn.refine(3, "layer", "sensitivity", x, verbose=True)

Authors:       Niklas Kochdumper, Tobias Ladner
Written:       23-November-2022 (polish)
Last update:   23-November-2022 (polish)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import List, Optional

def refine(self, max_order: int, type_: str, method: str, x: Optional[np.ndarray] = None, 
           verbose: bool = False, force_bounds: Optional[List[int]] = None, gamma: Optional[float] = None):
    """
    Refine the network using the maximum error bound
    
    Args:
        max_order: Maximum order for refinement
        type_: "all", "naive", "layer"- or "neuron"-wise refinement
        method: Refinement heuristic
           "approx_error", "sensitivity", "both", "random", "all", "layer_bias"
        x: Point used for sensitivity analysis
        verbose: Whether additional information should be displayed
        force_bounds: Orders at which to re-compute bounds
        gamma: Threshold for neuron-wise refinement
    """
    # parse input
    if force_bounds is None:
        force_bounds = []
    if gamma is None:
        gamma = 0.1
    
    # validate parameters
    if not isinstance(max_order, int):
        raise ValueError("max_order must be an integer")
    if type_ not in ['all', 'layer', 'neuron', 'naive']:
        raise ValueError("type_ must be one of: 'all', 'layer', 'neuron', 'naive'")
    if method not in ['all', 'random', 'approx_error', 'sensitivity', 'both', 'layer_bias']:
        raise ValueError("method must be one of: 'all', 'random', 'approx_error', 'sensitivity', 'both', 'layer_bias'")
    if x is None and (method == "sensitivity" or method == "both"):
        raise ValueError("No point for sensitivity analysis provided.")
    
    # get refinable layers
    refinable_layers = self.getRefinableLayers()
    
    # prepare refinement heuristic
    if method in ["sensitivity", "both"]:
        # calculate sensitivity
        self.calcSensitivity(x)
        # iterate through all layers
        for i in range(len(refinable_layers)):
            layer_i = refinable_layers[i]
            if method == "sensitivity":
                # set sensitivity as heuristic
                layer_i.refine_heu = np.linalg.norm(layer_i.sensitivity, axis=0).reshape(-1, 1)
            elif method == "both":
                # combine with approx error (already stored in refine_heu)
                layer_i.refine_heu = layer_i.refine_heu * np.linalg.norm(layer_i.sensitivity, axis=0).reshape(-1, 1)
    elif method == "random":
        # randomly set refine_heu
        for i in range(len(refinable_layers)):
            layer_i = refinable_layers[i]
            layer_i.refine_heu = np.random.rand(layer_i.refine_heu.shape[0], 1)
    elif method == "layer_bias":
        # stronger bias to earlier layers
        for i in range(len(refinable_layers)):
            layer_i = refinable_layers[i]
            layer_i.refine_heu = layer_i.refine_heu / (i + 1)
    elif method == "all":
        # refine all layers
        if verbose:
            print("Setting type='all', as heuristic was set to 'all'!")
        type_ = "all"
    
    # --- ALL REFINEMENT ---
    if type_ == "all" or type_ == "naive":
        if verbose:
            print("Refining all neurons...")
        
        count = 0
        for i in range(len(refinable_layers)):
            layer_i = refinable_layers[i]
            order_i = layer_i.order
            count += np.sum(order_i + 1 <= max_order)
            layer_i.order = np.minimum(order_i + 1, max_order)
        
        if verbose:
            print(f"Refined {count} neurons!")
    
    # --- LAYER-WISE REFINEMENT ---
    elif type_ == "layer":
        # determine most sensible choice for refinement
        heu = np.zeros((len(refinable_layers), 3))
        for i in range(len(refinable_layers)):
            layer_i = refinable_layers[i]
            # reduce with norm (TODO try max?)
            heu[i, 0] = np.linalg.norm(layer_i.refine_heu)
            heu[i, 1] = i  # layer i
            heu[i, 2] = np.max(layer_i.order)  # order in layer i
        
        # filter and sort
        heu = heu[heu[:, 2] < max_order, :]
        heu = heu[heu[:, 0] > 0, :]
        heu = heu[heu[:, 0].argsort()][::-1]  # sort by first column descending
        
        if heu.shape[0] > 0:
            # refine
            for i in range(heu.shape[0]):
                layer_i = refinable_layers[int(heu[i, 1])]
                order_i = layer_i.order
                if order_i < max_order:
                    if verbose:
                        print(f"Refined layer {int(heu[i, 1])} from order {np.max(order_i)} to {np.max(order_i) + 1}!")
                    order_i = order_i + 1
                    layer_i.order = order_i
                    
                    if np.max(order_i) in force_bounds:
                        # force re-calculation of bounds in all following layers
                        for j in range(int(heu[i, 1]) + 1, len(refinable_layers)):
                            layer_j = refinable_layers[j]
                            layer_j.l = []
                            layer_j.u = []
                    break
        else:
            if verbose:
                print(f"No layers are left to refine! Either max_order={max_order} reached or not refineable.")
    
    # --- NEURON-WISE REFINEMENT ---
    elif type_ == "neuron":
        # determine most sensible choice for refinement
        heu = np.zeros((0, 4))
        for i in range(len(refinable_layers)):
            layer_i = refinable_layers[i]
            heu_i = layer_i.refine_heu
            l = heu_i.shape[0]
            
            heu_i_new = np.zeros((l, 4))
            heu_i_new[:, 0] = heu_i.flatten()  # heuristic value
            heu_i_new[:, 1] = i  # layer i
            heu_i_new[:, 2] = np.arange(l)  # neuron in layer i
            heu_i_new[:, 3] = layer_i.order  # order in layer i
            
            heu = np.vstack([heu, heu_i_new]) if heu.size > 0 else heu_i_new
        
        # filter and sort
        heu = heu[heu[:, 3] < max_order, :]
        heu = heu[heu[:, 0] > 0, :]
        heu = heu[heu[:, 0].argsort()][::-1]  # sort by first column descending
        
        if heu.shape[0] > 0:
            M_max = heu[0, 0]
            heu = heu[heu[:, 0] > gamma * M_max, :]
            
            l_max = heu[0, 1]
            heu = heu[heu[:, 1] == l_max, :]
            
            # refine
            for i in range(heu.shape[0]):
                layer_i = refinable_layers[int(heu[i, 1])]
                order_i = layer_i.order[int(heu[i, 2])]
                M = heu[i, 0]
                
                if verbose:
                    print(f"Refined neuron {int(heu[i, 2])} from layer {int(heu[i, 1])} from order {order_i} to {order_i + 1}!")
                
                order_i = order_i + 1
                layer_i.order[int(heu[i, 2])] = order_i
                
                if order_i in force_bounds:
                    # force re-calculate bounds in all following layers
                    for j in range(int(heu[i, 1]) + 1, len(refinable_layers)):
                        layer_j = refinable_layers[j]
                        layer_j.l = []
                        layer_j.u = []
        else:
            if verbose:
                print(f"No neurons are left to refine! Either max_order={max_order} reached or not refineable.")
