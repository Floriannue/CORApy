"""
backpropagation - backpropagation function for syntax trees

This is a standalone function that calls the backpropagation method
on syntax tree objects.

Syntax:
    res = backpropagation(synTree, value, int)

Inputs:
    synTree - syntax tree object
    value - target interval value
    int - current interval domain

Outputs:
    res - contracted interval domain

Authors:       Matthias Althoff, Niklas Kochdumper
Written:       04-November-2019
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.interval.interval import Interval
    from .syntaxTree import SyntaxTree


def backpropagation(synTree: 'SyntaxTree', value: 'Interval', int_: 'Interval') -> 'Interval':
    """
    Backpropagation function for syntax trees
    
    Args:
        synTree: syntax tree object
        value: target interval value
        int_: current interval domain
        
    Returns:
        res: contracted interval domain
    """
    return synTree.backpropagation(value, int_)

