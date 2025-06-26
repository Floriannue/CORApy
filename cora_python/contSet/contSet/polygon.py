from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def polygon(S, *varargin):
    """
    convert a set to a (outer-approximative) polygon

    Syntax:
        pgon = polygon(S)

    Inputs:
        S - contSet object
        varargin - additonal parameters for vertices computation

    Outputs:
        pgon - polygon object
    """
    from cora_python.contSet.polygon import Polygon
    # check dimension of set
    if S.dim() != 2:
        raise CORAerror('CORA:wrongValue', 'first', 'Given set must be 2-dimensional.')

    # compute vertices
    V = S.vertices(*varargin)

    # init polygon
    pgon = Polygon(V)

    try:
        if S.representsa('convexSet'):
            pgon = pgon.convHull_()
    except Exception:
        # not always implemented / hard to determine for some sets
        # keep pgon as is
        pass
        
    return pgon 