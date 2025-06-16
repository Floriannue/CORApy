"""
modelChecking - check if a reachable set satisfies an STL formula

This function checks if a reachable set satisfies a Signal Temporal Logic (STL) formula
using various algorithms.

Authors: Niklas Kochdumper, Benedikt Seidl, Florian Lercher (MATLAB)
         Python translation by AI Assistant
Written: 09-November-2022 (MATLAB)
Python translation: 2025
"""

from typing import Any, Optional


def modelChecking(R, eq, alg: str = 'sampledTime', *args, **kwargs) -> bool:
    """
    Check if a reachable set satisfies an STL formula
    
    Args:
        R: ReachSet object
        eq: STL formula object
        alg: Algorithm used ('sampledTime', 'rtl', 'signals', 'incremental')
        *args: Additional arguments for specific algorithms
        **kwargs: Additional keyword arguments
        
    Returns:
        bool: True if formula is satisfied, False otherwise
        
    Example:
        # Create STL formula: x = stl('x',2); eq = until(x(2) < -0.7, x(1) > 0.7, interval(0,2))
        # sys = linearSys([0 -1; 1 0],[0;0])
        # params.R0 = zonotope([0;-1]); params.tFinal = 2
        # options.timeStep = 0.5; options.zonotopeOrder = 10; options.taylorTerms = 10
        # R = reach(sys,params,options)
        # res = modelChecking(R,eq)
    """
    # Validate inputs
    if not hasattr(R, 'timePoint') and not hasattr(R, 'timeInterval'):
        raise ValueError("R must be a reachSet object")
    
    # Check if STL formula object has required methods
    if not hasattr(eq, 'evaluate') and not hasattr(eq, 'check'):
        raise ValueError("eq must be an STL formula object")
    
    # Validate algorithm choice
    valid_algorithms = ['sampledTime', 'rtl', 'signals', 'incremental']
    if alg not in valid_algorithms:
        raise ValueError(f"Algorithm must be one of {valid_algorithms}")
    
    # Import private methods
    from .private.priv_modelCheckingSampledTime import priv_modelCheckingSampledTime
    from .private.priv_modelCheckingRTL import priv_modelCheckingRTL
    from .private.priv_modelCheckingSignals import priv_modelCheckingSignals
    from .private.priv_modelCheckingIncremental import priv_modelCheckingIncremental
    
    # Call the selected model checking algorithm
    if alg == 'sampledTime':
        return priv_modelCheckingSampledTime(R, eq)
    elif alg == 'rtl':
        return priv_modelCheckingRTL(R, eq)
    elif alg == 'signals':
        return priv_modelCheckingSignals(R, eq, **kwargs)
    elif alg == 'incremental':
        return priv_modelCheckingIncremental(R, eq, **kwargs) 