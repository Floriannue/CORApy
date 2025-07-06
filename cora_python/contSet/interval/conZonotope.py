
def conZonotope(I):
    """
    Converts an interval object into a conZonotope object.
    """
    from cora_python.contSet.conZonotope.conZonotope import ConZonotope
    # First, convert the interval to a Zonotope using the interval's zonotope method
    z = I.zonotope()
    
    # Then, convert the Zonotope to a ConZonotope
    cz = ConZonotope(z)
    
    return cz 