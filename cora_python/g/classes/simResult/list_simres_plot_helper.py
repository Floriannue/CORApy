#!/usr/bin/env python3

"""
Helper to make lists of SimResult objects behave like they have plot methods
This allows simRes.plot() to work when simRes is a list returned by simulateRandom()
"""

from typing import List, Any
from cora_python.g.classes.simResult.plot import plot as simres_plot


class SimResultList(list):
    """A list that behaves like a SimResult for plotting"""
    
    def plot(self, *args, **kwargs):
        """Plot method for list of SimResult objects"""
        return simres_plot(self, *args, **kwargs)
    
    def plotOverTime(self, *args, **kwargs):
        """PlotOverTime method for list of SimResult objects"""
        from cora_python.g.classes.simResult.plotOverTime import plotOverTime
        return plotOverTime(self, *args, **kwargs)


def make_simres_list_plottable(simres_list: List[Any]) -> SimResultList:
    """Convert a regular list of SimResult objects to a plottable list"""
    return SimResultList(simres_list) 