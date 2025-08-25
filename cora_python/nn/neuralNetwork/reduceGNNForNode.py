"""
reduceGNNForNode - removes all nodes that are not relevant to predict the
  node level output for the given node 

Syntax:
    gnn_red = reduceGNNForNode(obj,G,n0)

Inputs:
    obj - object of class (graph) neuralNetwork
    n0 - numeric, node of interest
    G - graph

Outputs:
    gnn_red - cell array

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner
Written:       21-March-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any, List, Union
import numpy as np
from .neuralNetwork import NeuralNetwork


def reduceGNNForNode(obj: NeuralNetwork, n0: int, G: Any) -> NeuralNetwork:
    """
    Remove all nodes that are not relevant to predict the node level output for the given node.
    
    Args:
        obj: NeuralNetwork object
        n0: Node of interest
        G: Graph object
        
    Returns:
        Reduced neural network
    """
    # Parse input
    if not isinstance(obj, NeuralNetwork):
        raise ValueError("First argument must be a NeuralNetwork")
    if not isinstance(n0, (int, np.integer)) or n0 <= 0:
        raise ValueError("Second argument must be a positive integer")
    if not hasattr(G, 'numnodes'):
        raise ValueError("Third argument must be a graph object")
    
    numNodes = G.numnodes
    if n0 > numNodes:
        raise ValueError("n0 has to be part of G")
    
    # Initialize
    numMPsteps = obj.getNumMessagePassingSteps()
    if numMPsteps == 0:
        # No GNN, return
        return obj
    
    # Initialize new layers
    layers = obj.layers
    layers_red = [None] * (len(layers) + numMPsteps + 1)
    cnt = 0
    
    # Compute distance of all neighbors
    # Note: This is a simplified implementation - in practice, you'd need
    # a proper graph library like NetworkX for the nearest() function
    try:
        # Try to use NetworkX if available
        import networkx as nx
        if hasattr(G, 'to_networkx'):
            G_nx = G.to_networkx()
        else:
            # Create a simple graph representation
            G_nx = nx.Graph()
            for i in range(numNodes):
                G_nx.add_node(i)
        
        # Compute shortest paths from n0 to all other nodes
        distances = nx.single_source_shortest_path_length(G_nx, n0 - 1)  # Convert to 0-based
        
        # Sort nodes by distance
        sorted_nodes = sorted(distances.items(), key=lambda x: x[1])
        remNeighbors = [n + 1 for n, _ in sorted_nodes]  # Convert back to 1-based
        distVec = [d for _, d in sorted_nodes]
        
    except ImportError:
        # Fallback: assume all nodes are reachable
        remNeighbors = list(range(numNodes))  # 0-based indexing like Python
        distVec = [0] + [1] * (numNodes - 1)
    
    # Add the node of interest at the beginning
    if n0 not in remNeighbors:
        remNeighbors.insert(0, n0)
        distVec.insert(0, 0)
    
    # Initial node reduction
    remMPsteps = numMPsteps
    idx_keep = [remNeighbors[i] for i in range(len(remNeighbors)) if distVec[i] <= remMPsteps + 1]
    
    # Create projection layer (simplified)
    from .layers.other.nnGNNProjectionLayer import nnGNNProjectionLayer
    layers_red[cnt] = nnGNNProjectionLayer(idx_keep, numNodes)
    remNeighbors = list(range(len(idx_keep)))  # 0-based indexing like Python
    distVec = [distVec[i] for i in range(len(remNeighbors)) if distVec[i] <= remMPsteps + 1]
    cnt += 1
    
    # Iterate over network
    for k in range(len(layers)):
        layer_k = layers[k]
        
        # Transfer to new layers
        if hasattr(layer_k, 'copy'):
            layers_red[cnt] = layer_k.copy()
        else:
            # Fallback: create a new instance
            layers_red[cnt] = layer_k
        cnt += 1
        
        # Check if it's a GCN layer
        if hasattr(layer_k, '__class__') and 'nnGCNLayer' in str(layer_k.__class__):
            remMPsteps -= 1
            
            # Reduce nodes
            if remMPsteps == 0:
                # Last MP step computed, only keep node of interest
                idx_keep = [remNeighbors[i] for i in range(len(remNeighbors)) if distVec[i] == 0]
            else:
                # Keep all neighbors in MP + 1 due to normalization
                idx_keep = [remNeighbors[i] for i in range(len(remNeighbors)) if distVec[i] <= remMPsteps + 1]
            
            layers_red[cnt] = nnGNNProjectionLayer(idx_keep, len(remNeighbors))
            remNeighbors = list(range(len(idx_keep)))  # 0-based indexing like Python
            distVec = [distVec[i] for i in range(len(remNeighbors)) if distVec[i] <= remMPsteps + 1]
            cnt += 1
    
    # Remove None entries
    layers_red = layers_red[:cnt]
    
    # Create reduced network
    gnn_red = NeuralNetwork(layers_red)
    
    return gnn_red
