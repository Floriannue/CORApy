"""
generateRandom - Generates a random zonotope

Syntax:
    Z = zonotope.generateRandom()
    Z = zonotope.generateRandom('Dimension',n)
    Z = zonotope.generateRandom('Dimension',n,'NrGenerators',nrGens)

Inputs:
    Name-Value pairs (all options, arbitrary order):
       <'Dimension',n> - dimension
       <'Center',c> - center
       <'NrGenerators',nrGens> - number of generators
       <'Distribution',type> - distribution for generators
           typeDist has to be {'uniform', 'exp', 'gamma'}

Outputs:
    Z - random zonotope

Example: 
    Z1 = zonotope.generateRandom();
    Z2 = zonotope.generateRandom('Dimension',3);
    Z3 = zonotope.generateRandom('Center',ones(2,1));
    Z4 = zonotope.generateRandom('Dimension',4,'NrGenerators',10);
    Z5 = zonotope.generateRandom('Distribution','gamma');

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Matthias Althoff, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       17-September-2019 (MATLAB)
Last update:   19-May-2022 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Optional, Union, Dict, Any
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .zonotope import Zonotope


def generateRandom(**kwargs) -> Zonotope:
    """
    Generates a random zonotope.
    
    Args:
        **kwargs: Name-value pairs:
            - Dimension: dimension
            - Center: center
            - NrGenerators: number of generators
            - Distribution: distribution for generators ('uniform', 'exp', 'gamma')
        
    Returns:
        Zonotope: Random zonotope
        
    Raises:
        CORAerror: If inputs are invalid or computation fails
    """
    # Check valid name-value pairs
    valid_keys = {'Dimension', 'Center', 'NrGenerators', 'Distribution'}
    for key in kwargs:
        if key not in valid_keys:
            raise CORAerror('CORA:wrongValue', 'name-value pair', f'Unknown parameter: {key}')
    
    # Extract parameters
    n = kwargs.get('Dimension', None)
    c = kwargs.get('Center', None)
    nrGens = kwargs.get('NrGenerators', None)
    type_dist = kwargs.get('Distribution', None)
    
    # Default computation for dimension
    if n is None:
        if c is None:
            nmax = 10
            n = np.random.randint(1, nmax + 1)
        else:
            n = len(c)
    
    # Default computation for center
    if c is None:
        c = 10 * np.random.randn(n, 1)
    else:
        c = np.asarray(c).reshape(-1, 1)
    
    # Default number of generators
    if nrGens is None:
        nrGens = 2 * n
    
    # Default distribution
    if type_dist is None:
        type_dist = 'uniform'
    
    # Generate random vector for the generator lengths
    l = np.zeros(nrGens)
    
    # Uniform distribution
    if type_dist == 'uniform':
        l = np.random.rand(nrGens)
    
    # Exponential distribution
    elif type_dist == 'exp':
        l = np.random.exponential(1, nrGens)
    
    # Gamma distribution
    elif type_dist == 'gamma':
        l = np.random.gamma(2, 1, nrGens)
    
    else:
        raise CORAerror('CORA:wrongValue', 'Distribution', 
                       f"Unknown distribution type: {type_dist}")
    
    # Init generator matrix
    G = np.zeros((n, nrGens))
    
    # Create generators
    for i in range(nrGens):
        # Generate random point on sphere
        gTmp = _randomPointOnSphere(n)
        # Stretch by length
        G[:, i] = l[i] * gTmp
    
    # Instantiate zonotope
    return Zonotope(c, G)


def _randomPointOnSphere(n: int) -> np.ndarray:
    """
    Generate a random point on the unit sphere in n dimensions.
    
    Args:
        n: Dimension of the space
        
    Returns:
        np.ndarray: Random point on the unit sphere
    """
    # Generate random vector from standard normal distribution
    x = np.random.randn(n)
    
    # Normalize to get a point on the unit sphere
    return x / np.linalg.norm(x) 