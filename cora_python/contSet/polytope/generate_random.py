"""
generateRandom - Generates a random non-empty polytope

Syntax:
    P_out = Polytope.generate_random()
    P_out = Polytope.generate_random('Dimension',n)
    P_out = Polytope.generate_random('Dimension',n,'NrConstraints',nrCon)
    P_out = Polytope.generate_random('Dimension',n,'NrConstraints',nrCon,
           'IsDegenerate',isDeg)
    P_out = Polytope.generate_random('Dimension',n,'NrConstraints',nrCon,
           'IsDegenerate',isDeg,'IsBounded',isBounded)

Inputs:
    Name-Value pairs (all options, arbitrary order):
       <'Dimension',n> - dimension
       <'NrConstraints',nrCon> - number of constraints
       <'IsDegenerate',isDeg> - degeneracy (true/false)
       <'IsBounded',isBounded> - boundedness (true/false)

Outputs:
    P_out - random polytope

Example: 
    P = Polytope.generate_random('Dimension',2);

Authors:       Niklas Kochdumper, Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       05-May-2020 (MATLAB)
Last update:   10-November-2022 (MW, adapt to new name-value pair syntax)
               30-November-2022 (MW, new algorithm)
               12-December-2022 (MW, less randomness for more stability)
Last revision: 14-July-2024 (MW, refactor)
Python translation: 2025
"""

import numpy as np
from typing import Union, Tuple, List, Any, Optional
from cora_python.g.functions.matlab.validate.check import checkNameValuePairs
from cora_python.g.functions.matlab.validate.preprocessing import readNameValuePair
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .polytope import Polytope


def generate_random(*varargin) -> Polytope:
    """
    Generates a random non-empty polytope
    restrictions:
    - number of constraints >= 1    =>  can be unbounded
    - number of constraints >= 2    =>  + can be unbounded and degenerate
    - number of constraints >= n+1  =>  + can be bounded
    - number of constraints >= n+2  =>  + can be bounded and degenerate
    """
    # name-value pairs -> number of input arguments is always a multiple of 2
    if len(varargin) % 2 != 0:
        raise CORAerror('CORA:evenNumberInputArgs')
    else:
        # read input arguments
        NVpairs = list(varargin)
        # check list of name-value pairs
        checkNameValuePairs(NVpairs, ['Dimension', 'NrConstraints', 'IsDegenerate', 'IsBounded'])
        
        # dimension given?
        NVpairs, n = readNameValuePair(NVpairs, 'Dimension', lambda x: isinstance(x, (int, np.integer)) and x > 0)
        # number of constraints given?
        NVpairs, nr_con = readNameValuePair(NVpairs, 'NrConstraints', lambda x: isinstance(x, (int, np.integer)) and x > 0)
        # degeneracy given?
        NVpairs, is_deg = readNameValuePair(NVpairs, 'IsDegenerate', lambda x: isinstance(x, bool))
        # boundedness given?
        NVpairs, is_bnd = readNameValuePair(NVpairs, 'IsBounded', lambda x: isinstance(x, bool))

    # quick exits (errors)
    aux_check_consistency(n, nr_con, is_deg, is_bnd)

    # set default values: depends on which parameters are given and their values
    n, nr_con, is_deg, is_bnd = aux_set_default_values(n, nr_con, is_deg, is_bnd)

    if not is_bnd:
        # idea for unbounded case: sample directions
        # degenerate case: one pair of constraints needs to be of the form
        #    a*x <= b   a*x >= b

        if n == 1 and not is_deg:
            A, b, Ae, be = aux_1d_unbounded_nondeg(n, nr_con)
        elif n == 2 and not is_deg:
            A, b, Ae, be = aux_2d_unbounded_nondeg(n, nr_con)
        elif n == 2 and is_deg:
            A, b, Ae, be = aux_2d_unbounded_deg(n, nr_con)
        else:
            A, b, Ae, be = aux_nd_unbounded(n, nr_con, is_deg)
        
    else:
        # bounded case: generate simplex and add constraints
        # it is ensured that nr_con > n+1 (non-degenerate)
        #                    nr_con > n+2 (degenerate)

        if n == 1 and is_deg:
            A, b, Ae, be = aux_1d_bounded_deg(n, nr_con)
        else:
            A, b, Ae, be = aux_nd_bounded(n, nr_con, is_deg)

    # instantiate polytope
    P_out = Polytope(A, b, Ae, be)

    # set properties (if they exist)
    if hasattr(P_out, 'bounded'):
        P_out.bounded.val = is_bnd
    if hasattr(P_out, 'fullDim'):
        P_out.fullDim.val = not is_deg
    if hasattr(P_out, 'emptySet'):
        P_out.emptySet.val = False

    return P_out


# Auxiliary functions -----------------------------------------------------

def aux_check_consistency(n: Optional[int], nr_con: Optional[int], is_deg: Optional[bool], is_bnd: Optional[bool]):
    """
    Checks for consistency between the given arguments for dimension, number
    of constraints, degeneracy, and boundedness
    """
    if nr_con is not None:
        if nr_con == 1:
            # only one constraint -> cannot be bounded or degenerate
            if is_bnd is not None and is_bnd:
                raise CORAerror('CORA:wrongValue', 'name-value pair isBounded',
                              'cannot be true if only one constraint given.')
            elif is_deg is not None and is_deg:
                raise CORAerror('CORA:wrongValue', 'name-value pair isDegenerate',
                              'cannot be true if only one constraint given.')
        
        if n is not None:
            # dimension given -> cannot be bounded unless nr_con >= n+1
            if is_bnd is not None and is_bnd and nr_con < n+1:
                raise CORAerror('CORA:wrongValue', 'name-value pair isBounded',
                              'cannot be true if number of constraints < dimension + 1')
    
    if (n is not None and n == 1 and is_bnd is not None and not is_bnd 
        and is_deg is not None and is_deg):
        # one-dimensional polytopes cannot be degenerate and unbounded
        raise CORAerror('CORA:wrongValue', 'name-value pair isDegenerate/isBounded',
                      'one-dimensional polytopes cannot be unbounded and degenerate at the same time')


def aux_set_default_values(n: Optional[int], nr_con: Optional[int], 
                          is_deg: Optional[bool], is_bnd: Optional[bool]) -> Tuple[int, int, bool, bool]:
    """
    Sets the default values for dimension, number of constraints, degeneracy,
    and boundedness according to the user-provided information
    """
    if n is None:
        if nr_con is not None and (is_bnd is None or is_bnd):
            # number of constraints given and boundedness either true by
            # default or because set so by user)
            if is_deg is not None and is_deg:
                # degenerate: n <= number of constraints - 1
                n = np.random.randint(1, nr_con-1)
            else:
                # non-degenerate (default): n <= number of constraints - 2
                n = np.random.randint(1, nr_con)
        else:
            # either: number of constraints not given
            # or:     number of constraints given, but unbounded
            # -> dimension can be set to any random value
            maxdim = 10
            if is_bnd is not None and not is_bnd and is_deg is not None and is_deg:
                # a set can only be unbounded and degenerate for n > 1
                n = 2
            else:
                n = np.random.randint(1, maxdim + 1)
    else:
        # dimension is given
        if nr_con is not None:
            # number of constraints given: decide boundedness
            if is_bnd is not None and is_bnd:
                # user wants the polytope to be bounded
                if is_deg is not None and is_deg:
                    # user wants the polytope to be degenerate
                    if nr_con < n+2:
                        raise CORAerror('CORA:wrongValue', 'name-value pair isDegenerate',
                                      'has to be false if the polytope should be bounded and the number of constraints < dimension + 2.')
                else:
                    # only bounded, degeneracy does not matter
                    if nr_con < n+1:
                        raise CORAerror('CORA:wrongValue', 'name-value pair isBounded',
                                      'has to be false if number of constraints < dimension + 1.')
            else:
                # either boundedness not defined by user or set to false
                if is_bnd is None:
                    # set to true/false depending on nr_con < n+1?
                    if nr_con < n+1:
                        is_bnd = False
                    else:
                        is_bnd = True
                elif nr_con < n+1:
                    # ensure that nr_con >= n+1
                    raise CORAerror('CORA:wrongValue', 'name-value pair isBounded',
                                  'has to be false if number of constraints < dimension + 1.')

    # default degeneracy
    if is_deg is None:
        is_deg = False

    # default boundedness
    if is_bnd is None:
        is_bnd = True

    # default computation for number of constraints
    if nr_con is None:
        if n == 1 and is_deg:
            nr_con = 2
        else:
            nr_con = 2*n + np.random.randint(n//2, n + 1)
    else:
        # number of constraints are given -> ensure that other provided values
        # are admissible
        if is_bnd is not None and is_bnd and nr_con < n+1:
            raise CORAerror('CORA:wrongValue', 'name-value pair IsBounded',
                          'cannot be true if number of constraints < n+1.')

    return n, nr_con, is_deg, is_bnd


def aux_1d_unbounded_nondeg(n: int, nr_con: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random unbounded, non-degenerate polytope in 1D"""
    # 1D algorithm: can only be bounded at one side (towards +/-Inf)
    s = np.sign(np.random.randn())  # randn will practically never be 0...
    if s == 0:  # just in case
        s = 1
    A = s * np.ones((nr_con, n))
    b = s * np.random.rand(nr_con, 1)

    # no equality constraints
    Ae = np.zeros((0, 1))
    be = np.array([]).reshape(-1, 1)

    return A, b, Ae, be


def aux_2d_unbounded_nondeg(n: int, nr_con: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random unbounded, non-degenerate polytope in 2D"""
    # 2D: sample directions from one half of the plane, offset > 0
    A = np.zeros((nr_con, n))
    b = np.zeros((nr_con, 1))
    for i in range(nr_con):
        A[i, :] = [np.random.randn(), np.random.rand()]
        b[i, 0] = np.random.rand()

    # map by random matrix
    U, _, _ = np.linalg.svd(np.random.randn(n, n))  # invertible
    A = A @ np.linalg.inv(U)

    # no equality constraints
    Ae = np.zeros((0, 2))
    be = np.array([]).reshape(-1, 1)

    return A, b, Ae, be


def aux_2d_unbounded_deg(n: int, nr_con: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random unbounded, degenerate polytope in 2D"""
    # 2D: unbounded and degenerate -> line (only allows for constraints
    # along the same parallel vector, fill up with redundancies until
    # number of constraints reached)

    # random vector
    vec = np.random.randn(n, 1)
    # normalize
    vec = vec / np.linalg.norm(vec)

    # random offset
    offset = np.random.randn()

    # init constraint matrix and offset
    A = np.zeros((nr_con, n))
    b = np.zeros((nr_con, 1))
    
    A[0, :] = vec.flatten()
    A[1, :] = -vec.flatten()
    b[0, 0] = offset
    b[1, 0] = -offset

    # add redundant halfspace
    for i in range(2, nr_con):
        s = np.sign(np.random.randn())
        if s == 0:
            s = 1
        A[i, :] = s * vec.flatten()
        b[i, 0] = s * offset + np.random.rand()

    # map by random matrix
    U, _, _ = np.linalg.svd(np.random.randn(n, n))  # invertible
    A = A @ np.linalg.inv(U)

    # no equality constraints
    Ae = np.zeros((0, 2))
    be = np.array([]).reshape(-1, 1)

    return A, b, Ae, be


def aux_nd_unbounded(n: int, nr_con: int, is_deg: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random unbounded, potentially degenerate, polytope in nD"""
    # all higher-dimensional cases: sample directions, but keep one
    # entry always 0 -> unboundness in one dimension ensured

    # init constraint matrix and offset in dimension n-1
    tempdirs = np.random.randn(nr_con, n-1)
    norms = np.linalg.norm(tempdirs, axis=1, keepdims=True)
    A_temp = tempdirs / norms
    b = np.ones((nr_con, 1))

    # randomly selected dimension
    rand_dim = np.random.randint(0, n)

    # set all entries in that dimension to 0
    A = np.zeros((nr_con, n))
    A[:, :rand_dim] = A_temp[:, :rand_dim]
    A[:, rand_dim+1:] = A_temp[:, rand_dim:]

    # degenerate case
    if is_deg:
        # adapt last one using second-to-last one
        A[-2, :] = -A[-1, :]
        b[-2, 0] = -b[-1, 0]

        # permutate matrix, offset
        order = np.random.permutation(nr_con)
        A = A[order, :]
        b = b[order, :]

    # map by random matrix
    U, _, _ = np.linalg.svd(np.random.randn(n, n))  # invertible
    A = A @ np.linalg.inv(U)

    # no equality constraints
    Ae = np.zeros((0, n))
    be = np.array([]).reshape(-1, 1)

    return A, b, Ae, be


def aux_1d_bounded_deg(n: int, nr_con: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random bounded, degenerate polytope in 1D"""
    # has to be a point; add redundant constraints until number of constraints
    point = np.random.randn()
    A = np.zeros((nr_con, n))
    b = np.zeros((nr_con, 1))
    
    A[0, 0] = 1
    A[1, 0] = -1
    b[0, 0] = point
    b[1, 0] = -point
    
    for i in range(2, nr_con):
        A[i, 0] = 1
        b[i, 0] = np.random.rand()

    # no equality constraints
    Ae = np.zeros((0, n))
    be = np.array([]).reshape(-1, 1)

    return A, b, Ae, be


def aux_nd_bounded(n: int, nr_con: int, is_deg: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random bounded, potentially degenerate polytope in nD"""
    # step 1: simplex
    nr_con_simplex = n + 1
    A = np.zeros((nr_con, n))
    b = np.ones((nr_con, 1))
    
    # Identity part
    A[:n, :n] = np.eye(n)
    # Last constraint of simplex
    A[n, :] = -np.ones(n) / np.sqrt(n)

    # step 2: sample random directions of length 1, then the resulting
    #         halfspace is always non-redundant
    for i in range(nr_con_simplex, nr_con):
        # random direction
        tempdir = np.random.randn(n)
        A[i, :] = tempdir / np.linalg.norm(tempdir)
        # set offset to 1 to ensure non-redundancy
        b[i, 0] = 1

    # degenerate case
    if is_deg:
        # permutate matrix, offset
        order = np.random.permutation(nr_con)
        A = A[order, :]
        b = b[order, :]

        # adapt last one using second-to-last one
        A[-2, :] = -A[-1, :]
        b[-2, 0] = -b[-1, 0]

    # step 3: rotate
    U, _, _ = np.linalg.svd(np.random.randn(n, n))  # invertible
    A = A @ np.linalg.inv(U)

    # step 4: translate polytope by some small offset
    z = np.random.randn(n, 1)
    b = b + A @ z

    # no equality constraints
    Ae = np.zeros((0, n))
    be = np.array([]).reshape(-1, 1)

    return A, b, Ae, be 