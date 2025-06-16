"""
plotAsGraph - plot branches of reachable set as graph

This function plots branches of reachable set as graph, branches with only
one time-point solution (due to an instant outgoing transition) are
marked in red.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 07-June-2023 (MATLAB)
Python translation: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import warnings

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    warnings.warn("NetworkX not available. plotAsGraph functionality will be limited.")


def plotAsGraph(R) -> Optional[object]:
    """
    Plot branches of reachable set as graph
    
    Branches with only one time-point solution (due to an instant outgoing 
    transition) are marked in red.
    
    Args:
        R: ReachSet object or list of ReachSet objects
        
    Returns:
        Handle to graph object (or None if no output requested)
    """
    if not HAS_NETWORKX:
        warnings.warn("NetworkX not available. Cannot plot as graph.")
        return None
    
    if not isinstance(R, list):
        R = [R]
    
    if len(R) == 1:
        # Single node graph
        G = nx.Graph()
        G.add_node(0)
        
        # Plot the graph
        fig, ax = plt.subplots(figsize=(6, 4))
        pos = {0: (0, 0)}
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='blue', 
                node_size=500, font_size=16, font_weight='bold')
        ax.set_title('ReachSet Graph')
        
        return fig
    
    else:
        # Convert reachSet object to directed graph
        edge_start = []
        edge_end = []
        
        for i in range(1, len(R)):
            if R[i].parent > 0:
                edge_start.append(R[i].parent - 1)  # Convert to 0-based indexing
                edge_end.append(i)
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(len(R)):
            G.add_node(i)
        
        # Add edges
        for start, end in zip(edge_start, edge_end):
            G.add_edge(start, end)
        
        # Node labels are locations
        node_labels = {}
        node_colors = []
        
        for i, reach_set in enumerate(R):
            node_labels[i] = f"[{reach_set.loc}]"
            
            # Check if this is an instant transition
            # (reachSet object with only one time-point solution)
            is_instant = (not reach_set.timeInterval or 
                         len(reach_set.timeInterval.get('set', [])) == 0) and \
                        (reach_set.timePoint and 
                         len(reach_set.timePoint.get('set', [])) == 1)
            
            # Use red for instant transitions, blue for others
            node_colors.append('red' if is_instant else 'blue')
        
        # Plot the graph
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use hierarchical layout if possible
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            # Fallback to spring layout
            pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw the graph
        nx.draw(G, pos, ax=ax, labels=node_labels, node_color=node_colors,
                node_size=1000, font_size=10, font_weight='bold',
                arrows=True, arrowsize=20, edge_color='gray')
        
        ax.set_title('ReachSet Graph\n(Red: Instant Transitions, Blue: Regular)')
        
        return fig 