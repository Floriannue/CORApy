import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.preprocessing import setDefaultValues
from cora_python.g.functions.matlab.validate.check import inputArgsCheck

def plotRandPoint(S, *varargin):
    """
    Plots a point cloud of random points from inside a set

    Syntax:
        han = plotRandPoint(S)
        han = plotRandPoint(S,dims,N,type)

    Inputs:
        S - contSet object
        dims - (optional) dimensions that should be projected
        N - (optional) number of points for the point cloud
        type - (optional) plot type

    Outputs:
        han - handle for the resulting graphics object
    """

    # default values for the optional input arguments
    # In python, dimension indices start at 0
    defaults = [[0, 1], 1000, '.k']
    dims, N, type_ = setDefaultValues(defaults, varargin)

    # input argument check
    inputArgsCheck([[S, 'att', 'contSet'],
                    [dims, 'att', ['numeric'], ['integer', 'nonnegative', 'vector']],
                    [N, 'att', ['numeric'], ['integer', 'nonnegative', 'scalar']],
                    [type_, 'str', []]])

    # check if dimension to be plotted is not too high
    if len(dims) > 3:
        raise CORAerror('CORA:plotProperties', 'Number of dimensions to plot has to be <= 3.')

    # generate N random points inside the set
    points = S.randPoint(N)
    
    # Ensure dims is a list or array of integers
    if not isinstance(dims, (list, np.ndarray)):
        dims = [dims]
    dims = np.array(dims)
    
    # plot the point cloud projected onto desired dimensions
    if dims.ndim == 1 and len(dims) == 1:
        han = plt.plot(points[dims[0], :], np.zeros_like(points[dims[0], :]), type_)[0]
    elif len(dims) == 2:
        han = plt.plot(points[dims[0], :], points[dims[1], :], type_)[0]
    elif len(dims) == 3:
        ax = plt.gca()
        if not isinstance(ax, Axes3D):
            # If the current axes are not 3D, create a new 3D subplot
            fig = plt.gcf()
            ax = fig.add_subplot(111, projection='3d')
        han = ax.plot(points[dims[0], :], points[dims[1], :], points[dims[2], :], type_)[0]
    else:
        raise CORAerror('CORA:wrongValue', 'second', 'Number of dimensions to plot has to be <= 3.')

    return han 