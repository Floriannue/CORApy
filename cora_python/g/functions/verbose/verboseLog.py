"""
verboseLog - verbose logging function for CORA algorithms

This function provides verbose output for debugging and monitoring
the progress of CORA algorithms, especially reachability analysis.

Syntax:
    verboseLog(verbose, k, t, tStart, tFinal)

Inputs:
    verbose - verbose level (0 = no output, 1+ = increasing verbosity)
    k - step counter
    t - current time
    tStart - start time (optional)
    tFinal - final time (optional)

Outputs:
    (none - prints to console)

Example:
    verboseLog(1, 5, 0.5, 0.0, 2.0)

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 17-June-2016 (MATLAB)
Last update: 25-July-2016 (MATLAB, restructure params/options)
Python translation: 2025
"""

import sys
from typing import Union, Optional


def verboseLog(verbose: int, k: Optional[int] = None, t: Optional[float] = None, 
               tStart: Optional[float] = None, tFinal: Optional[float] = None):
    """
    Verbose logging function for CORA algorithms
    
    Args:
        verbose: Verbosity level (0 = no output, higher = more verbose)
        k: Step counter (optional)
        t: Current time (optional) 
        tStart: Start time (optional)
        tFinal: Final time (optional)
    """
    
    if verbose == 0:
        return
    
    # Basic step information
    if k is not None and t is not None:
        if verbose == 1:
            # Simple progress output
            if tFinal is not None and tStart is not None:
                progress = (t - tStart) / (tFinal - tStart) * 100
                print(f"Step {k:4d}: t = {t:8.4f} ({progress:5.1f}%)")
            else:
                print(f"Step {k:4d}: t = {t:8.4f}")
        
        elif verbose >= 2:
            # Detailed output
            if tFinal is not None and tStart is not None:
                progress = (t - tStart) / (tFinal - tStart) * 100
                remaining = tFinal - t
                print(f"Step {k:4d}: t = {t:8.4f} ({progress:5.1f}%), remaining = {remaining:8.4f}")
            else:
                print(f"Step {k:4d}: t = {t:8.4f}")
            
            # Additional debug info for higher verbosity
            if verbose >= 3:
                print(f"         tStart = {tStart}, tFinal = {tFinal}")
    
    elif k is not None:
        # Just step counter
        print(f"Step {k}")
    
    elif t is not None:
        # Just time
        print(f"t = {t:8.4f}")
    
    # Flush output for immediate display
    sys.stdout.flush()


def verboseLogReach(verbose: int, k: int, t: float, tStart: float, tFinal: float,
                   timeStep: Optional[float] = None, error: Optional[float] = None):
    """
    Specialized verbose logging for reachability analysis
    
    Args:
        verbose: Verbosity level
        k: Step counter
        t: Current time
        tStart: Start time
        tFinal: Final time
        timeStep: Current time step size (optional)
        error: Current error (optional)
    """
    
    if verbose == 0:
        return
    
    progress = (t - tStart) / (tFinal - tStart) * 100
    remaining = tFinal - t
    
    if verbose == 1:
        print(f"Step {k:4d}: t = {t:8.4f} ({progress:5.1f}%)")
    
    elif verbose >= 2:
        if timeStep is not None:
            print(f"Step {k:4d}: t = {t:8.4f} ({progress:5.1f}%), dt = {timeStep:8.4f}, remaining = {remaining:8.4f}")
        else:
            print(f"Step {k:4d}: t = {t:8.4f} ({progress:5.1f}%), remaining = {remaining:8.4f}")
        
        if verbose >= 3 and error is not None:
            print(f"         error = {error:10.6e}")
    
    sys.stdout.flush()


def verboseLogAdaptive(verbose: int, k: int, k_iter: int, t: float, tStart: float, tFinal: float,
                      timeStep: float, error: Optional[float] = None, 
                      error_bound: Optional[float] = None):
    """
    Specialized verbose logging for adaptive reachability analysis
    
    Args:
        verbose: Verbosity level
        k: Step counter
        k_iter: Iteration counter within step
        t: Current time
        tStart: Start time
        tFinal: Final time
        timeStep: Current time step size
        error: Current error (optional)
        error_bound: Error bound (optional)
    """
    
    if verbose == 0:
        return
    
    progress = (t - tStart) / (tFinal - tStart) * 100
    
    if verbose == 1:
        print(f"Step {k:4d}.{k_iter}: t = {t:8.4f} ({progress:5.1f}%), dt = {timeStep:8.4f}")
    
    elif verbose >= 2:
        remaining = tFinal - t
        print(f"Step {k:4d}.{k_iter}: t = {t:8.4f} ({progress:5.1f}%), dt = {timeStep:8.4f}, remaining = {remaining:8.4f}")
        
        if verbose >= 3:
            if error is not None and error_bound is not None:
                error_ratio = error / error_bound if error_bound > 0 else float('inf')
                print(f"         error = {error:10.6e}, bound = {error_bound:10.6e}, ratio = {error_ratio:6.2f}")
            elif error is not None:
                print(f"         error = {error:10.6e}")
    
    sys.stdout.flush()


def verboseLogHeader(verbose: int, algorithm: str = "", system_name: str = ""):
    """
    Print header information for verbose logging
    
    Args:
        verbose: Verbosity level
        algorithm: Name of the algorithm
        system_name: Name of the system
    """
    
    if verbose == 0:
        return
    
    print("=" * 60)
    if algorithm and system_name:
        print(f"CORA - {algorithm} for system '{system_name}'")
    elif algorithm:
        print(f"CORA - {algorithm}")
    else:
        print("CORA - Reachability Analysis")
    print("=" * 60)
    
    sys.stdout.flush()


def verboseLogFooter(verbose: int, computation_time: Optional[float] = None):
    """
    Print footer information for verbose logging
    
    Args:
        verbose: Verbosity level
        computation_time: Total computation time in seconds (optional)
    """
    
    if verbose == 0:
        return
    
    print("=" * 60)
    if computation_time is not None:
        print(f"Computation completed in {computation_time:.3f} seconds")
    else:
        print("Computation completed")
    print("=" * 60)
    
    sys.stdout.flush() 