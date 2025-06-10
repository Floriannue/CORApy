"""
generateRandom - Generates a random interval

Syntax:
    I = interval.generateRandom()
    I = interval.generateRandom('Dimension',n)
    I = interval.generateRandom('Dimension',n,'Center',c)
    I = interval.generateRandom('Dimension',n,'Center',c,'MaxRadius',r)

Inputs:
    Name-Value pairs (all options, arbitrary order):
       <'Dimension',n> - dimension
       <'Center',c> - center
       <'MaxRadius',r> - maximum radius for each dimension or scalar

Outputs:
    I - random interval

Example: 
    I = interval.generateRandom('Dimension',2);
    plot(I);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Mark Wetzlinger, Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       17-September-2019
Last update:   19-May-2022 (MW, name-value pair syntax)
               23-February-2023 (MW, add 'Center' and 'MaxRadius')
               22-May-2023 (TL, bugfix: all dimensions had same radius)
Last revision: ---
"""

import numpy as np
from .interval import Interval

def generateRandom(**kwargs):
    """
    Generates a random interval.
    
    Args:
        **kwargs: keyword arguments including:
            Dimension (int or tuple): dimension of the interval
            Center (array): center of the interval
            MaxRadius (float or array): maximum radius for each dimension
            
    Returns:
        Interval: randomly generated interval
    """
    # Extract named arguments
    n = kwargs.get('Dimension', None)
    c = kwargs.get('Center', None)
    r = kwargs.get('MaxRadius', None)
    
    # Parse dimension argument
    if n is not None:
        if np.isscalar(n):
            # Rewrite as (n, 1) for easier handling of matrices
            n = (int(n), 1)
        else:
            n = tuple(n)
    
    # Default computation of dimension
    if n is None:
        if c is not None:
            # Center given -> read out dimension
            c = np.asarray(c)
            n = c.shape
        elif r is not None and not np.isscalar(r):
            # Radius given with dimension specs
            r = np.asarray(r)
            n = r.shape
        else:
            # Generate random dimension
            nmax = 10
            n = (np.random.randint(1, nmax + 1), 1)
    
    # Validation checks
    if c is not None:
        c = np.asarray(c)
        if c.shape != n:
            raise ValueError("Center has to match the dimension")
    
    if r is not None and not np.isscalar(r):
        r = np.asarray(r)
        if r.shape != n:
            raise ValueError("MaxRadius has to match the dimension")
    
    # Default computation of center
    if c is None:
        # Set somewhere in the neighborhood of the origin
        c = -2 + 4 * np.random.rand(*n)
    
    # Default computation of maximum radius
    if r is None:
        r = 10 * np.random.rand(*n)
    
    # Convert scalar radius to array if needed
    if np.isscalar(r):
        r = r * np.ones(n)
    
    # Effective radius of interval (has to be symmetric to maintain center)
    rad = r / 2 * np.random.rand(*n)
    
    # Instantiate interval
    return Interval(c - rad, c + rad) 