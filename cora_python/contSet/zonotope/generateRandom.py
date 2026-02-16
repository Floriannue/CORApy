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


Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Matthias Althoff, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       17-September-2019 (MATLAB)
Last update:   19-May-2022 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import Optional, Union, Dict, Any, List
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.helper.sets.contSet.zonotope.randomPointOnSphere import randomPointOnSphere
from .zonotope import Zonotope


def generateRandom(**kwargs) -> Zonotope:
    """
    Generates a random zonotope.

    Syntax (Python-style lowercase kwargs, consistent with LinearSys.generateRandom):
        Z = Zonotope.generateRandom()
        Z = Zonotope.generateRandom(dimension=2)
        Z = Zonotope.generateRandom(dimension=2, nr_generators=5)

    Args:
        **kwargs: dimension, center, nr_generators, distribution (all optional).

    Returns:
        Zonotope: Random zonotope
    """
    n = kwargs.get('dimension')
    c = kwargs.get('center')
    nrGens = kwargs.get('nr_generators')
    type_dist = kwargs.get('distribution')
    if kwargs.keys() - {'dimension', 'center', 'nr_generators', 'distribution'}:
        raise CORAerror('CORA:wrongValue', 'kwargs', 'Only dimension, center, nr_generators, distribution allowed')
    
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
        # Ensure center is a column vector
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
        gTmp = randomPointOnSphere(n)
        # Stretch by length
        G[:, i] = (l[i] * gTmp).flatten()
    
    # Instantiate zonotope
    return Zonotope(c, G) 