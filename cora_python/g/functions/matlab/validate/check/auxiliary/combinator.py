"""
combinator - Perform basic permutation and combination samplings

This function supports multiple modes:
- PERMUTATIONS WITH REPETITION/REPLACEMENT: combinator(N,K,'p','r')  --  N >= 1, K >= 0
- PERMUTATIONS WITHOUT REPETITION/REPLACEMENT: combinator(N,K,'p')  --  N >= 1, N >= K >= 0
- COMBINATIONS WITH REPETITION/REPLACEMENT: combinator(N,K,'c','r')  --  N >= 1, K >= 0
- COMBINATIONS WITHOUT REPETITION/REPLACEMENT: combinator(N,K,'c')  --  N >= 1, N >= K >= 0

Syntax:
    A = combinator(N, K)
    A = combinator(N, K, s1)
    A = combinator(N, K, s1, s2)

Inputs:
    N - positive integer (upper bound of set 1:N)
    K - non-negative integer (number of elements to choose)
    s1 - first mode string ('p' for permutations, 'c' for combinations)
    s2 - second mode string ('r' for repetition, or any other string for no repetition)

Outputs:
    A - matrix where each row contains one combination/permutation

Example:
    A = combinator(4, 2, 'c')  # Combinations without repetition
    A = combinator(4, 2, 'p', 'r')  # Permutations with repetition

Authors: Matt Fig (MATLAB)
         Python translation by AI Assistant
Written: 5/30/2009 (MATLAB)
Python translation: 2025
"""

import numpy as np
from itertools import combinations, permutations, product, combinations_with_replacement


def combinator(N: int, K: int, *varargin) -> np.ndarray:
    """
    Compute combinations or permutations of the set 1:N taken K at a time
    
    Args:
        N: Upper bound of set (positive integer)
        K: Number of elements to choose (non-negative integer)
        *varargin: Optional mode strings
            - If 2 args: defaults to 'p','r' (permutations with repetition)
            - If 3 args: s1='p' or 'c', s2 defaults to 'n' (no repetition)
            - If 4 args: s1='p' or 'c', s2='r' or other (repetition or not)
        
    Returns:
        Array where each row contains one combination/permutation (1-indexed)
    """
    
    # MATLAB: ng = nargin;
    ng = len(varargin) + 2  # N and K are always present
    
    # MATLAB: if ng == 2
    if ng == 2:
        # MATLAB: s1 = 'p';
        # MATLAB: s2 = 'r';
        s1 = 'p'
        s2 = 'r'
    # MATLAB: elseif ng == 3
    elif ng == 3:
        s1 = varargin[0]
        # MATLAB: s2 = 'n';
        s2 = 'n'
    # MATLAB: elseif ng ~= 4
    elif ng != 4:
        raise ValueError('Only 2, 3 or 4 inputs are allowed.  See help.')
    else:
        s1 = varargin[0]
        s2 = varargin[1]
    
    # MATLAB: if isempty(N) || K == 0
    if (isinstance(N, (list, np.ndarray)) and len(N) == 0) or K == 0:
        # MATLAB: A = [];
        return np.array([], dtype=int).reshape(0, K)
    
    # MATLAB: elseif numel(N)~=1 || N<=0 || ~isreal(N) || floor(N) ~= N
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError('N should be one real, positive integer. See help.')
    # MATLAB: elseif numel(K)~=1 || K<0 || ~isreal(K) || floor(K) ~= K
    if not isinstance(K, (int, np.integer)) or K < 0:
        raise ValueError('K should be one real non-negative integer. See help.')
    
    # MATLAB: STR = lower(s1(1)); % We are only interested in the first letter.
    STR = s1[0].lower() if len(s1) > 0 else 'c'
    
    # MATLAB: if ~strcmpi(s2(1),'r')
    if len(s2) == 0 or s2[0].lower() != 'r':
        # MATLAB: STR = [STR,'n'];
        STR = STR + 'n'
    else:
        # MATLAB: STR = [STR,'r'];
        STR = STR + 'r'
    
    # MATLAB: switch STR
    if STR == 'pr':
        # MATLAB: A = perms_rep(N,K);
        A = perms_rep(N, K)
    elif STR == 'pn':
        # MATLAB: A = perms_no_rep(N,K);
        A = perms_no_rep(N, K)
    elif STR == 'cr':
        # MATLAB: A = combs_rep(N,K);
        A = combs_rep(N, K)
    elif STR == 'cn':
        # MATLAB: A = combs_no_rep(N,K);
        A = combs_no_rep(N, K)
    else:
        raise ValueError('Unknown option passed.  See help')
    
    return A


def perms_rep(N: int, K: int) -> np.ndarray:
    """
    Permutations with repetition/replacement
    pr = @(N,K) N^K;  Number of rows.
    """
    if N == 1:
        return np.ones((1, K), dtype=int)
    elif K == 1:
        return np.arange(1, N + 1, dtype=int).reshape(-1, 1)
    
    # Generate all permutations with repetition using product
    result = list(product(range(1, N + 1), repeat=K))
    return np.array(result, dtype=int)


def perms_no_rep(N: int, K: int) -> np.ndarray:
    """
    Permutations without repetition/replacement
    pn = @(N,K) prod(1:N)/(prod(1:(N-K)));  Number of rows.
    """
    if K > N:
        raise ValueError('When no repetitions are allowed, K must be less than or equal to N')
    if N == K:
        # All permutations
        result = list(permutations(range(1, N + 1), K))
        return np.array(result, dtype=int)
    elif K == 1:
        return np.arange(1, N + 1, dtype=int).reshape(-1, 1)
    
    # Generate permutations without repetition
    result = list(permutations(range(1, N + 1), K))
    return np.array(result, dtype=int)


def combs_rep(N: int, K: int) -> np.ndarray:
    """
    Combinations with repetition/replacement (multichoose)
    cr = @(N,K) prod((N):(N+K-1))/(prod(1:K)); Number of rows.
    """
    # Generate combinations with replacement
    result = list(combinations_with_replacement(range(1, N + 1), K))
    return np.array(result, dtype=int)


def combs_no_rep(N: int, K: int) -> np.ndarray:
    """
    Combinations without repetition/replacement
    cn = @(N,K) prod(1:N)/(prod(1:K)*prod(1:(N-K)));  Number of rows.
    """
    if K > N:
        raise ValueError('When no repetitions are allowed, K must be less than or equal to N')
    
    # Handle edge cases
    if K == 0:
        return np.empty((1, 0), dtype=int)
    if N == 0:
        return np.empty((0, K), dtype=int)
    
    # Generate combinations without repetition
    result = list(combinations(range(1, N + 1), K))
    if len(result) > 0:
        return np.array(result, dtype=int)
    else:
        return np.empty((0, K), dtype=int) 