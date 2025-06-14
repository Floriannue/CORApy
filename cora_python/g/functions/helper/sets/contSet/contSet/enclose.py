"""
enclose - compute convex hull (enclosure) of two sets

This function computes the convex hull of two sets, which encloses both sets.

Syntax:
    result = enclose(set1, set2)

Inputs:
    set1 - first set
    set2 - second set

Outputs:
    result - convex hull of the two sets

Authors: Python translation by AI Assistant
Written: 2025
"""

import numpy as np

def enclose(set1, set2):
    """
    Compute convex hull (enclosure) of two sets
    
    Args:
        set1: First set
        set2: Second set
        
    Returns:
        Convex hull of the two sets
    """
    # Check if sets have a convHull method
    if hasattr(set1, 'convHull'):
        return set1.convHull(set2)
    elif hasattr(set2, 'convHull'):
        return set2.convHull(set1)
    
    # Fallback: use interval hull
    from cora_python.contSet.interval import Interval
    
    # Convert to intervals using their interval() method if available
    if hasattr(set1, 'interval'):
        int1 = set1.interval()
    elif isinstance(set1, Interval):
        int1 = set1
    else:
        # Convert using Interval constructor if possible
        try:
            int1 = Interval(set1)
        except:
            # Last resort: use Minkowski sum as approximation
            return set1 + set2
        
    if hasattr(set2, 'interval'):
        int2 = set2.interval()
    elif isinstance(set2, Interval):
        int2 = set2
    else:
        # Convert using Interval constructor if possible
        try:
            int2 = Interval(set2)
        except:
            # Last resort: use Minkowski sum as approximation
            return set1 + set2
    
    # Compute interval hull (union of intervals)
    inf_result = np.minimum(int1.inf, int2.inf)
    sup_result = np.maximum(int1.sup, int2.sup)
    
    return Interval(inf_result, sup_result) 